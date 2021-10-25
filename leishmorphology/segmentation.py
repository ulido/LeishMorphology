from pathlib import Path
from PIL import Image
import tifffile
import json
import numpy as np
import functools
from tqdm.auto import tqdm

from skimage.filters import unsharp_mask, meijering, threshold_otsu, threshold_li
from skimage.morphology import closing, erosion, dilation, binary_closing, binary_erosion, binary_dilation, diameter_closing, medial_axis, remove_small_objects, skeletonize, medial_axis, disk, opening
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.transform import rotate
from skimage.restoration import rolling_ball

from scipy import ndimage
from scipy.spatial import distance_matrix

from .datastructures import SegmentedCell, SegmentedCellCollection
from .moore_neighborhood import moore_neighborhood

def unsharp_mask_multi(img, radius, amount):
    """Apply an unsharp mask `N` times where `N` is the number of radii in the list `radius` and the number of amounts in the list `amount`. If `amount` is a scalar it will be promoted to a list of length `N`. Returns the resulting sharpened image.
    """
    # Promote amount to a list
    try:
        iter(amount)
    except TypeError:
        amount = len(radius)*[amount]

    if len(radius) != len(amount):
        raise ArgumentError("The number of radiusses have to be the same as the number of amounts.")
    
    # Iterate over radii and amounts, repeatedly applied skimage's unsharp_mask.
    uns = img
    for r, a in zip(radius, amount):
        uns = unsharp_mask(uns, radius=r, amount=a)
    
    return uns

def image_thumbnail(θ, intensity_img, cell_coords, flagella_coords, width=400, height=400, overlay_color=np.array([1, 1, 0]), flagella_overlay_color=np.array([1,0,1]), alpha=0.3):
    """Create a thumbnail with the correct orientation (horizontal cell, flagellum pointing to the right) and the cell
    highlighted.
    """
    
    # Calculate the bounding box
    coords = np.concatenate([cell_coords, flagella_coords], axis=0)
    x = coords[:,0].min()
    y = coords[:,1].min()
    xx = coords[:,0].max()
    yy = coords[:,1].max()
    
    # We need to cut out a bigger image if possible if we need to rotate later.
    rwidth = int(2**0.5*width+1)
    rheight = int(2**0.5*height+1)
    
    # Get the center of the bounding box
    cx, cy = (xx + x) // 2, (yy + y) // 2
    
    # Constrain the width or height if we are close to the image boundary
    w = rwidth if cx > rwidth/2 else 2*cx
    h = rheight if cy > rheight/2 else 2*cy

    # Shift the global coordinates to thumbnail local
    offset = np.array([[cx-w//2, cy-h//2]])
    cell_coords -= offset
    flagella_coords -= offset
    
    # Get the thumbnail and rotate such that the flagellum points right
    img = intensity_img[cx-w//2 : cx + w//2, cy - h//2 : cy + h//2].astype(float)
    img = (img-img.min())/(img.max()-img.min())
    
    # Crop coordinates to our thumbnail size
    # This should never really be triggered, it's a safety.
    cell_coords = cell_coords[(cell_coords < [img.shape]).all(axis=1)]
    flagella_coords = flagella_coords[(flagella_coords < [img.shape]).all(axis=1)]

    # Dilate the flagella mask a bit, it's only a skeleton and would not be visible otherwise.
    flag_mask = np.zeros_like(img, dtype=bool)
    flag_mask[flagella_coords[:,0], flagella_coords[:,1]] = True
    flag_mask = binary_dilation(flag_mask, selem=disk(radius=3))
    overlay_labels = np.zeros(img.shape, dtype=int)
    overlay_labels[cell_coords[:,0],cell_coords[:,1]] = 1
    overlay_labels[flag_mask] = 2
    rgb = label2rgb(overlay_labels, image=img, colors=[overlay_color, flagella_overlay_color], bg_label=0, image_alpha=1.0)
    #rgb[cell_coords[:,0], cell_coords[:,1]] = alpha*overlay_color[np.newaxis] + (1-alpha)*rgb[cell_coords[:,0], cell_coords[:,1]]
    #rgb[flag_mask] = alpha*flagella_overlay_color[np.newaxis,:] + (1-alpha)*rgb[flag_mask]

    # Rotate such that the flagellum points right.
    rgb = rotate(rgb, -θ + 90, resize=False)
    
    # Now cut it to the requested width/height if possible
    real_w, real_h = rgb.shape[:2]
    cut_w, cut_h = np.clip([real_w//2-width//2, real_h//2-height//2], 0, None)
    rgb = rgb[cut_w:-cut_w-1, cut_h:-cut_h-1]
    if (np.array(rgb.shape) == 0).any():
        return None
    
    # Need to make it a uint8, otherwise Pillow will complain.
    return (255*rgb).astype(np.uint8)

def _filter_region_property(mask, filter_func, intensity_image=None):
    mask = mask.copy()
    for r in regionprops(label(mask), intensity_image=intensity_image):
        if filter_func(r):
            mask[r.coords[:,0],r.coords[:,1]] = False
    return mask

def filter_touching_boundary(mask, offset):
    """Filter regions that are within `offset` of the image borders."""
    w, h = mask.shape
    def filter_func(r):
        x, y, xx, yy = r.bbox
        return (x<offset) or (y<offset) or (xx>=w-offset) or (yy>=h-offset)
    return _filter_region_property(mask, filter_func)

def filter_solidity(mask, solidity_threshold):
    """Filter regions by their solidity (i.e. the ratio between the mask area and its convex hull."""
    def filter_func(r):
        return r.solidity < solidity_threshold
    return _filter_region_property(mask, filter_func)

def filter_min_intensity(mask, intensity_image):
    """Filter regions by their minimum intensity value."""
    label_mask = label(mask)
    min_intensities = {r.label: r.intensity_image[r.image].min() for r in regionprops(label_mask, intensity_image=intensity_image)}
    labels, vals = np.array(list(min_intensities.keys())), np.array(list(min_intensities.values()))
    threshold = threshold_li(vals)
    mask = mask.copy()
    mask[np.isin(label_mask, labels[vals > threshold])] = False
    return mask

class ImageSet:
    def __init__(self, path, name=None,
                 unsharp_mask_settings={"radius": 20, "amount": 0.9},
                 phase_threshold_settings={"n_standard_deviations": 1},
                 threshold_closing_settings={"selem": disk(radius=3)}, #np.ones((5, 5))},
                 segmentation_filter_settings={"n_standard_deviations": 2, "area_bounds": (2000, 7000)},
                 rolling_ball_settings={"radius": 10, "n_standard_deviations": 1.5, "min_size": 2000},
                ):
        self.path = Path(path)
        if name is None:
            self.name = str(self.path)
        else:
            self.name = name
        self._phase_path, self._gene_path, self._dapi_path = sorted([x for x in (self.path / "Default").glob('img_channel???_position000_time000000000_z000.tif')])
        
        self._settings = {
            "unsharp_mask": unsharp_mask_settings,
            "phase_threshold": phase_threshold_settings,
            "threshold_closing": threshold_closing_settings,
            "segmentation_filter": segmentation_filter_settings,
            "rolling_ball": rolling_ball_settings,
        }
        
    @functools.cached_property
    def metadata(self):
        with tifffile.TiffFile(self._phase_path, 'rb') as tif:
            return json.loads(tif.imagej_metadata["Info"])
    
    @property
    def pixelsize(self):
        return self.metadata["PixelSizeUm"]
    
    @functools.cached_property
    def phase_image(self):
        with tifffile.TiffFile(self._phase_path, 'rb') as tif:
            return tif.asarray()
    
    @functools.cached_property
    def gene_image(self):
        with tifffile.TiffFile(self._gene_path, 'rb') as tif:
            return tif.asarray()
    
    @functools.cached_property
    def dapi_image(self):
        with tifffile.TiffFile(self._dapi_path, 'rb') as tif:
            return tif.asarray()

    @functools.cached_property
    def sharpened_phase_image(self):
        return unsharp_mask(self.phase_image, **self._settings["unsharp_mask"])
    
    @functools.cached_property
    def segmented_cell_bodies(self):
        # Morphological grayscale opening with small disk (highlights out-of-focus shadows)
        open1 = opening(self.sharpened_phase_image, selem=disk(5))
        # Morphological grayscale opening with larger disk (highlights mostly cell bodies)
        open2 = opening(self.sharpened_phase_image, selem=disk(15))
        # Threshold both using Otsu. The small disk leads to a binary image with zeros at the shadows.
        thresh1 = open1 < threshold_otsu(open1)
        thresh2 = open2 < threshold_otsu(open2)
        # Binary AND the two thresholded images - this gives better shadow rejection than the large disk alone.
        # We also remove small objects here.
        thresh = remove_small_objects(thresh1 & thresh2, min_size=1000)
        # Filter regions that are within 10 pixels of the image border.
        thresh = filter_touching_boundary(thresh, 10)
        # Close holes and get rid of small thin structures (such as the flagella).
        thresh = binary_dilation(binary_dilation(binary_erosion(binary_erosion(thresh, selem=disk(2)), selem=disk(2)), selem=disk(2)), selem=disk(2))
        # Remove small disconnected objects again
        thresh = remove_small_objects(thresh, min_size=1000)
        
        # Filter by min intensity of the large disk opening. In focus cells are very dark in this image.
        mask = filter_min_intensity(thresh, open2)
        # Finally, filter by solidity (that's the ratio between the object area (pixel count) and the
        # object's convex hull area).
        mask = filter_solidity(mask, 0.7)
        # Label and return.
        return label(mask)
    
    @functools.cached_property
    def meijering_neuriteness(self):
        # Meijering neuriteness is a Hessian based filter for long thin structures.
        return meijering(self.sharpened_phase_image, sigmas=range(1, 6, 1), black_ridges=True)

    @functools.cached_property
    def labelled_cells_flagella(self):
        # Threshold the meijering filter image
        meij = self.meijering_neuriteness
        # Use the Li thresholding method (minimum entropy method)
        thresh = meij > threshold_li(meij)
        return label(thresh)
        
    @functools.cached_property
    def labelled_cells(self):
        # Perform morphological operations to close the cell bodies and remove flagella.
        # This is ultimately so that we can subtract the cell bodies from the flagella binary image.
        # Note that this is purely to find the flagella, not to segment the cell bodies, this is done in
        # `segmented_cell_bodies`.
        # THIS CAN LIKELY BE MORE STREAMLINED BUT WORKS FOR NOW.
        selem = disk(radius=4)
        closed = dilation(
            dilation(
                erosion(
                    erosion(
                        closing(255*(self.labelled_cells_flagella > 0), selem=selem),
                        selem=selem),
                    selem=selem),
                selem=selem),
            selem=selem) > 0
        
        # Make sure the cell bodies have the same labels as in `labelled_cells_flagella`.
        # This was originally to be able to associate flagella with cell bodies, but didn't work
        # very well. Abandoned this idea in favour of using a greedy distance based matching below.
        labelled = label(closed)
        for r in regionprops(labelled, intensity_image=self.labelled_cells_flagella):
            lbl, counts = np.unique(r.intensity_image, return_counts=True)
            counts = counts[lbl != 0]
            lbl = lbl[lbl != 0]
            lbl = lbl[counts.argmax()]
            labelled[r.coords[:,0], r.coords[:,1]] = lbl
        return labelled
    
    @functools.cached_property
    def labelled_flagella(self):
        # Subtract the cell bodies from the flagella threshold
        thresh_both = self.labelled_cells_flagella > 0
        thresh_cells = self.labelled_cells > 0
        thresh_flag = remove_small_objects(thresh_both ^ (thresh_both & thresh_cells), min_size=500)

        # Using medial axis skeletonization, filter out "flagella" that are wider than 8 pixels.
        # NOTE: THIS MIGHT BE TOO RESTRICTIVE! REVISIT!
        _, dist = medial_axis(thresh_flag, return_distance=True)
        return dist
        for r in regionprops(label(thresh_flag), intensity_image=dist):
            if r.intensity_image.max() > 12:
                thresh_flag[r.coords[:,0], r.coords[:,1]] = False
        # Make sure the flagella are consistently labelled the same as the cell bodies.
        return thresh_flag * self.labelled_cells_flagella
    
    @functools.cached_property
    def labelled_skeletonized_flagella(self):
        # Skeletonize the found flagella using medial axis transform.
        skel, dist = medial_axis(self.labelled_flagella > 0, return_distance=True)
        
        # Separate branches of the skeleton by finding and blanking points with a connectivity greater than 2
        # (i.e. three or more white pixels touching a given white pixel).
        connectivity_kernel = np.ones((3,3))
        separated_branches = skel*(ndimage.convolve(1*skel, connectivity_kernel, mode="constant", cval=0) < 4)
        # Again, filter by flagella width
        # THIS MIGHT NOT ACTUALLY BE NEEDED ANYMORE.
        for r in regionprops(label(separated_branches), intensity_image=dist):
            if r.intensity_image.max() > 8:
                separated_branches[r.coords[:,0], r.coords[:,1]] = 0
        
        # Remove small branches.
        return label(remove_small_objects(separated_branches, min_size=40, connectivity=2))
    
    @functools.cached_property
    def labelled_flagella_extrema(self):
        # Find the two extrema points of all flagella.
        # These are the two points with connectivity == 1 on each skeletonized flagellum.
        skel = self.labelled_skeletonized_flagella
        connectivity_kernel = np.ones((3,3))
        return skel*(ndimage.convolve(1*(skel>0), connectivity_kernel, mode="constant", cval=0) == 2)
    
    @functools.cached_property
    def match_flagella(self, angle_cost=50.0):
        # This function matches the extrema points of flagella to the posterior of cells.
        # This is done by identifying the kinetoplast in the DAPI channel (always to the posterior side of the cell centre).
        # We then project the fitted ellipse's major axis length from the cell centre outward in the direction of the
        # kinetoplast, this is the posterior point.
        # Remember the posterior points together with the cell labels.
        cell_points = []
        cell_labels = []
        for r in regionprops(self.segmented_cell_bodies):
            c = np.array(r.centroid)
            a = r.major_axis_length/2
            θ = r.orientation

            # THIS IS DUPLICATED FROM BELOW - NEED TO SPIN OUT INTO IT'S OWN PROPERTY
            bbox = np.array(r.bbox).reshape(2,2)
            # Find the maximum of the DAPI intensity in the region. This will be the location of the kinetoplast.
            kinetoplast_pos = np.unravel_index((self.dapi_image[r.slice]*r.image).argmax(), r.image.shape) + bbox[0]
            # The orientation of the kinetoplast relative to the center tells us the right/left flip of the cell
            ϕ = np.arctan2(*(kinetoplast_pos - c)[::-1])
            # Adjust the orientation angle such that if we later rotate the image, the flagellum will point right.
            if np.cos(θ)*np.cos(ϕ)+np.sin(θ)*np.sin(ϕ) < 0:
                θ += np.pi

            v = np.array([np.cos(θ), np.sin(θ)])
            x = c+a*v
            # We store the point and the direction of the cell.
            cell_points.append((x[0], x[1], v[0], v[1]))
            cell_labels.append(r.label)
        cell_points = np.array(cell_points)

        # Find the extrema points and the corresponding flagella labels, create a mapping between them. 
        c = np.array(np.where(self.labelled_flagella_extrema > 0)).T
        flag_labels = self.labelled_skeletonized_flagella[c[:,0], c[:,1]]
        flag_lbl_extrema = {lbl: c[flag_labels == lbl,:] for lbl in np.unique(flag_labels)}

        # Now iterate over the extrema points, finding the overall orientation of the given flagellum
        # and store all this in `flag_points`, together with the labels in `flag_labels`.
        flag_points = []
        flag_labels = []
        for lbl, extrema in flag_lbl_extrema.items():
            if extrema.shape[0] != 2:
                print(extrema)
                raise ValueError
            a, b = extrema
            d = b-a
            d = d/np.hypot(d[0], d[1])
            flag_points.append((a[0], a[1], d[0], d[1]))
            flag_points.append((b[0], b[1], -d[0], -d[1]))
            flag_labels.extend([lbl, lbl])
        flag_points = np.array(flag_points)

        # Distance matrix between each cell posterior point and each flagellum extrema.
        # Note that this isn't just the planar distance, but includes the distance of the direction vectors too.
        # This is a poor man's cost function to penalise differences in cell and flaqellum orientation.
        D = distance_matrix(cell_points[:,:2], flag_points[:,:2])
        D /= 40.0
        D[D > 1] = 1e5
        A = 1-(cell_points[:,np.newaxis,2:] * flag_points[np.newaxis,:,2:]).sum(axis=-1)
        C = D + A
        used_cell_indices = []
        used_flag_indices = []
        label_mapping = {}
        for i, j in zip(*np.unravel_index(C.argsort(axis=None), C.shape)):
            # We disallow distances beyond 50 (this includes the orientation mismatch cost).
            if (C[i,j] <= 2) & (i not in used_cell_indices) and (j not in used_flag_indices):
                used_cell_indices.append(i)
                used_flag_indices.append(j)
                label_mapping[cell_labels[i]] = (flag_labels[j], np.linalg.norm(np.array(cell_points[i][:2]) - flag_points[j][:2]))
        
        # Return the label mapping
        return label_mapping
    
    @functools.cached_property
    def identified_cells(self):
        # Create a new collection to hold the identified cells.
        cells = SegmentedCellCollection(self.name, self.path)
        pixelsize = self.pixelsize
        
        # Iterate over all segmented cell bodies, look if we have a matching flagellum,
        # create a nicely rotated thumbnail and store in the collection.
        flag_labels = self.labelled_skeletonized_flagella
        flag_extrema = self.labelled_flagella_extrema
        for r in tqdm(regionprops(self.segmented_cell_bodies)):
            # Include flagellum properties if we could match one
            try:
                # Lookup flagellum label
                flagella_label, flagella_distance = self.match_flagella[r.label]
                # Flagellum mask and coordinates
                flag_mask = flag_labels == flagella_label
                if not flag_mask.any():
                    raise IndexError
                fcoords = np.array(np.where(flag_mask)).T

                # Find the Moore neighborhood contour (i.e. the flagellum path)
                # This is a contour, so it traces the path twice, need to divide by two.
                # We also need to dilate the flagellum mask, otherwise the contour algorithm gets stuck.
                # SHOULD USE A SIMPLE CONTOUR ALGORITHM FOR THIS, BUT THIS WORKS FOR NOW EVEN IF IT IS SLOW.
                mmask = binary_dilation(np.pad(flag_mask, 1), selem=np.ones((2,2)))
                mn = moore_neighborhood(mmask)
                flag_path_length = (flagella_distance+np.linalg.norm(np.diff(mn, axis=0), axis=1).sum())*pixelsize/2
                # Straight length is simply the distance between the flagellum's extrema
                flag_straight_length = (flagella_distance+np.linalg.norm(
                    np.diff(np.array(np.where(flag_extrema == flagella_label)).T, axis=0)))*pixelsize
            except (KeyError, IndexError):
                # Found no flagella, give NaNs
                fcoords = np.empty((0, 2), dtype=int)
                flag_straight_length = np.nan
                flag_path_length = np.nan
            
            # Get the center of the region's bounding box
            bbox = np.array(r.bbox).reshape(2,2)
            c = bbox[0] + np.diff(bbox, axis=0)//2
            # Regionprops does fit an ellipse for us and gives us the major and minor axes, as well as the orientation angle.
            θ = r.orientation
            # Find the maximum of the DAPI intensity in the region. This will be the location of the kinetoplast.
            kinetoplast_pos = np.unravel_index((self.dapi_image[r.slice]*r.image).argmax(), r.image.shape) + bbox[0]
            # The orientation of the kinetoplast relative to the center tells us the right/left flip of the cell
            ϕ = np.arctan2(*(kinetoplast_pos - c)[0][::-1])
            # Adjust the orientation angle such that if we later rotate the image, the flagellum will point right.
            if np.cos(θ)*np.cos(ϕ)+np.sin(θ)*np.sin(ϕ) < 0:
                θ += np.pi

            # Create the thumbnail
            thumb = image_thumbnail(180*θ/np.pi, self.phase_image, r.coords, fcoords)
            
            cells.cell_list.append(SegmentedCell(
                length=r.major_axis_length*pixelsize,
                width=r.minor_axis_length*pixelsize,
                area=r.area*pixelsize**2,
                solidity=r.solidity,
                flagellum_path_length=flag_path_length,
                flagellum_straight_length=flag_straight_length,
                image=Image.fromarray(thumb),
                cell_body_coords=r.coords,
                flagellum_coords=fcoords,
                coordinates=(c[0,0],c[0,1]),
            ))

        return cells
