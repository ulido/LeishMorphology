from pathlib import Path
from PIL import Image
import tifffile
import json
import numpy as np
import functools
from tqdm.auto import tqdm

from skimage.filters import unsharp_mask, meijering
from skimage.morphology import closing, erosion, dilation, binary_closing, binary_erosion, binary_dilation, diameter_closing, medial_axis, remove_small_objects, skeletonize, medial_axis, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.transform import rotate
from skimage.restoration import rolling_ball

from scipy import ndimage
from scipy.spatial import distance_matrix

from .datastructures import SegmentedCell, SegmentedCellCollection

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

def image_thumbnail(θ, intensity_img, cell_coords, flagella_coords, width=400, height=400, overlay_color=np.array([255, 0, 0]), flagella_overlay_color=np.array([0,0,255]), alpha=0.3):
    """Create a thumbnail with the correct orientation (horizontal cell, flagellum pointing to the right) and the cell
    highlighted.
    """
    
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

    offset = np.array([[cx-w//2, cy-h//2]])
    cell_coords -= offset
    flagella_coords -= offset
    
    # Get the thumbnail and rotate such that the flagellum points right
    img = intensity_img[cx-w//2 : cx + w//2, cy - h//2 : cy + h//2]
    rgb = (np.array([255, 255, 255])[np.newaxis, np.newaxis]*((img-img.min())/(img.max()-img.min()))[:, :, np.newaxis])
    
    cell_coords = cell_coords[(cell_coords < [rgb.shape[:2]]).all(axis=1)]
    flagella_coords = flagella_coords[(flagella_coords < [rgb.shape[:2]]).all(axis=1)]

    rgb[cell_coords[:,0], cell_coords[:,1]] = alpha*overlay_color[np.newaxis] + (1-alpha)*rgb[cell_coords[:,0], cell_coords[:,1]]
    flag_mask = np.zeros_like(img, dtype=bool)
    flag_mask[flagella_coords[:,0], flagella_coords[:,1]] = True
    flag_mask = binary_dilation(flag_mask, selem=disk(radius=3))
    rgb[flag_mask] = alpha*flagella_overlay_color[np.newaxis,:] + (1-alpha)*rgb[flag_mask]

    rgb = rotate(rgb, -θ + 90, resize=False)
    
    # Now cut it to the requested width/height if possible
    real_w, real_h = rgb.shape[:2]
    cut_w, cut_h = np.clip([real_w//2-width//2, real_h//2-height//2], 0, None)
    rgb = rgb[cut_w:-cut_w-1, cut_h:-cut_h-1]
    if (np.array(rgb.shape) == 0).any():
        return None
    
    return rgb.astype(np.uint8)

class ImageSet:
    def __init__(self, path, name=None,
                 unsharp_mask_settings={"radius": range(1, 35, 5), "amount": 0.4},
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
        return unsharp_mask_multi(self.phase_image, **self._settings["unsharp_mask"])
    
    @functools.cached_property
    def rolling_ball_threshold(self):
        rb = rolling_ball(self.phase_image, radius=self._settings["rolling_ball"]["radius"])
        th = rb.mean() - self._settings["rolling_ball"]["n_standard_deviations"]*rb.std()
        thimg = rb < th
        return remove_small_objects(thimg, min_size=self._settings["rolling_ball"]["min_size"])
    
    @functools.cached_property
    def thresholded_phase_image(self):
        mean = np.mean(self.sharpened_phase_image)
        std = np.std(self.sharpened_phase_image)
        threshold = mean-self._settings["phase_threshold"]["n_standard_deviations"]*std
        return self.sharpened_phase_image < threshold
    
    @functools.cached_property
    def closed_thresholded_phase_image(self):
        selem = self._settings["threshold_closing"]["selem"]
#         ret = (
#             dilation(
#                 dilation(
#                     erosion(
# #                        erosion(
                            
#                             closing(255*self.thresholded_phase_image, selem=selem),
# #                            selem=selem),
#                         selem=selem),
#                     selem=selem),
#                 selem=selem))
        return dilation(erosion(diameter_closing(255*self.thresholded_phase_image, diameter_threshold=12), selem=selem), selem=selem)
    
    @functools.cached_property
    def labelled_threshold_image(self):
        return label(self.closed_thresholded_phase_image)
    
    @functools.cached_property
    def meijering_neuriteness(self):
        #uns = unsharp_mask_multi(self.phase_image, range(0, 5, 1), amount=0.4)
        return meijering(self.sharpened_phase_image, sigmas=range(1, 6, 1), black_ridges=True)

    @functools.cached_property
    def labelled_cells_flagella(self):
        meij = self.meijering_neuriteness
        th = meij.mean() + 0.5*meij.std()
        thresh = meij > th
        return label(thresh)
        
    @functools.cached_property
    def labelled_cells(self):
        selem = disk(radius=3) #np.ones((4, 4))
        closed = dilation(
            dilation(
                erosion(
                    erosion(
                        closing(255*(self.labelled_cells_flagella > 0), selem=selem),
                        selem=selem),
                    selem=selem),
                selem=selem),
            selem=selem) > 0
        labelled = label(closed)
        for r in regionprops(labelled, intensity_image=self.labelled_cells_flagella):
            lbl = np.unique(r.intensity_image)
            lbl = lbl[lbl != 0]
            if (lbl.shape[0] == 1):
                labelled[r.coords[:,0], r.coords[:,1]] = lbl[0]
            else:
                labelled[r.coords[:,0], r.coords[:,1]] = 0
        return labelled
    
    @functools.cached_property
    def labelled_flagella(self):
        thresh_both = self.labelled_cells_flagella > 0
        thresh_cells = self.labelled_cells > 0
        thresh_flag = remove_small_objects(thresh_both ^ (thresh_both & thresh_cells), min_size=500)
        _, dist = medial_axis(thresh_flag, return_distance=True)
        for r in regionprops(label(thresh_flag), intensity_image=dist):
            if r.intensity_image.max() > 8:
                thresh_flag[r.coords[:,0], r.coords[:,1]] = False
        return thresh_flag * self.labelled_cells_flagella
    
    @functools.cached_property
    def labelled_skeletonized_flagella(self):
        skel, dist = medial_axis(self.labelled_flagella > 0, return_distance=True)
        
        connectivity_kernel = np.ones((3,3))
        separated_branches = skel*(ndimage.convolve(1*skel, connectivity_kernel, mode="constant", cval=0) != 4)
        for r in regionprops(label(separated_branches), intensity_image=dist):
            if r.intensity_image.max() > 8:
                separated_branches[r.coords[:,0], r.coords[:,1]] = 0
        
        return label(remove_small_objects(separated_branches, min_size=40, connectivity=2))
    
    @functools.cached_property
    def labelled_flagella_extrema(self):
        skel = self.labelled_skeletonized_flagella
        connectivity_kernel = np.ones((3,3))
        return skel*(ndimage.convolve(1*(skel>0), connectivity_kernel, mode="constant", cval=0) == 2)
    
    @functools.cached_property
    def identified_cells_threshold_image(self):
        """Segment all separated regions in the binary image `bin_img` and label them uniquely. Filter the resulting regions
        according to three criteria:
            1) Regions cannot touch the image border.
            2) A region needs to have a pixel area bounded by `area_bounds`.
            3) The minimum intensity in the region (determined via the data given in `intensity_img`) needs to be less than
            the intensity threshold. This threshold is given by `mean(intensity_img)-intensity_stds*std(intensity_img)`.

        Optionally, also return the minimum intensities of all processed regions (also rejected ones) if `return_min_intensities`
        is `True`.
        """
        # Label each separated region.
        label_img = self.labelled_threshold_image.copy()
        intensity_img = self.phase_image

        # Calculate the minimum intensity threshold
        min_intensity_threshold = intensity_img.mean() - self._settings["segmentation_filter"]["n_standard_deviations"]*intensity_img.std()
        
        area_bounds = self._settings["segmentation_filter"]["area_bounds"]
        
        rb_threshed = self.rolling_ball_threshold

        # Iterate through all detected separate regions in the image, filtering those we do not want to see.
        min_intensities = []
        reject_reasons = {}
        for region in tqdm(regionprops(label_img, intensity_image=intensity_img)):
            reject = None
            c = np.array(region.bbox).reshape((2, 2)).T
            min_region_intensity = region.intensity_image[region.intensity_image > 0].min()
            min_intensities.append(min_region_intensity)

            # Do not consider the region if...
            if (c == 0).any() or (c == label_img.shape).any():
                # ... it touches any image border, or...
                reject = "border"
            elif (region.area < area_bounds[0]) or (region.area > area_bounds[1]):
                # ... it's area is too small or too large, or...
                reject = "area"
            elif min_region_intensity > min_intensity_threshold:
                # ... it's minimum intensity is above the mean minus (typically) two standard deviations of the full image.
                # This criterion gets rid of out of focus cells, because only in-focus cells have low enough intensity
                # somewhere in their interior in a phase contrast image. Out of focus cells do not achieve this due to
                # stray light.
                reject = "min_intensity"
            elif not rb_threshed[region.coords[:,0], region.coords[:,1]].any():
                reject = "rolling_ball"
            # elif region.solidity < 0.85:
            #     reject = "solidity"
            #elif nmf(region):
            #    reject = "normal_profile"

            # Remove the region from the binary mask if we want to forget it
            if reject is not None:
                label_img[label_img == region.label] = 0
            reject_reasons[region.label] = reject
        self._min_intensities = min_intensities
        self._reject_reasons = reject_reasons
        return label_img
    
    @functools.cached_property
    def match_flagella(self, angle_cost=100.0):
        cell_points = []
        cell_labels = []
        for r in regionprops(self.identified_cells_threshold_image):
            c = np.array(r.centroid)
            a = r.major_axis_length/2
            θ = r.orientation

            bbox = np.array(r.bbox).reshape(2,2)
            # Find the maximum of the DAPI intensity in the region. This will be the location of the kinetochore.
            kinetochore_pos = np.unravel_index((self.dapi_image[r.slice]*r.image).argmax(), r.image.shape) + bbox[0]
            # The orientation of the kinetochore relative to the center tells us the right/left flip of the cell
            ϕ = np.arctan2(*(kinetochore_pos - c)[::-1])
            # Adjust the orientation angle such that if we later rotate the image, the flagellum will point right.
            if np.cos(θ)*np.cos(ϕ)+np.sin(θ)*np.sin(ϕ) < 0:
                θ += np.pi

            v = np.array([np.cos(θ), np.sin(θ)])
            x = c+a*v
            cell_points.append((x[0], x[1], angle_cost*v[0], angle_cost*v[1]))
            cell_labels.append(r.label)
        cell_points = np.array(cell_points)

        c = np.array(np.where(self.labelled_flagella_extrema > 0)).T
        flag_labels = self.labelled_skeletonized_flagella[c[:,0], c[:,1]]
        flag_lbl_extrema = {lbl: c[flag_labels == lbl,:] for lbl in np.unique(flag_labels)}

        flag_points = []
        flag_labels = []
        for lbl, extrema in flag_lbl_extrema.items():
            if extrema.shape[0] != 2:
                raise ValueError
            a, b = extrema
            d = b-a
            d = angle_cost*d/np.hypot(d[0], d[1])
            flag_points.append((a[0], a[1], d[0], d[1]))
            flag_points.append((b[0], b[1], -d[0], -d[1]))
            flag_labels.extend([lbl, lbl])
        flag_points = np.array(flag_points)

        D = distance_matrix(cell_points[:,:], flag_points[:,:])
        used_cell_indices = []
        used_flag_indices = []
        label_mapping = {}
        for i, j in zip(*np.unravel_index(D.argsort(axis=None), D.shape)):
            if (D[i,j] < 50) & (i not in used_cell_indices) and (j not in used_flag_indices):
                used_cell_indices.append(i)
                used_flag_indices.append(j)
                label_mapping[cell_labels[i]] = flag_labels[j]
        
        return label_mapping
    
    @functools.cached_property
    def identified_cells(self):
        cells = SegmentedCellCollection(self.name, self.path)
        pixelsize = self.pixelsize
        flag_labels = self.labelled_skeletonized_flagella
        for r in tqdm(regionprops(self.identified_cells_threshold_image)):
            try:
                flagella_label = self.match_flagella[r.label]
                fcoords = np.array(np.where(flag_labels == flagella_label)).T
            except (KeyError, IndexError):
                fcoords = np.empty((0, 2), dtype=int)
            
            # Get the center of the region's bounding box
            bbox = np.array(r.bbox).reshape(2,2)
            c = bbox[0] + np.diff(bbox, axis=0)//2
            # Regionprops does fit an ellipse for us and gives us the major and minor axes, as well as the orientation angle.
            θ = r.orientation
            # Find the maximum of the DAPI intensity in the region. This will be the location of the kinetochore.
            kinetochore_pos = np.unravel_index((self.dapi_image[r.slice]*r.image).argmax(), r.image.shape) + bbox[0]
            # The orientation of the kinetochore relative to the center tells us the right/left flip of the cell
            ϕ = np.arctan2(*(kinetochore_pos - c)[0][::-1])
            # Adjust the orientation angle such that if we later rotate the image, the flagellum will point right.
            if np.cos(θ)*np.cos(ϕ)+np.sin(θ)*np.sin(ϕ) < 0:
                θ += np.pi

            thumb = image_thumbnail(180*θ/np.pi, self.phase_image, r.coords, fcoords)

            cells.cell_list.append(SegmentedCell(
                length=r.major_axis_length*pixelsize,
                width=r.minor_axis_length*pixelsize,
                area=r.area*pixelsize**2,
                solidity=r.solidity,
                image=Image.fromarray(thumb)))

        return cells
