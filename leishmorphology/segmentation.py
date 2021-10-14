from pathlib import Path
from PIL import Image
import tifffile
import json
import numpy as np
import functools
from tqdm.auto import tqdm

from skimage.filters import unsharp_mask, frangi
from skimage.morphology import closing, erosion, dilation, binary_closing, binary_erosion, binary_dilation, diameter_closing, medial_axis
from skimage.measure import label, regionprops, EllipseModel
from skimage.color import label2rgb
from skimage.transform import rotate

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

def image_thumbnail(bbox, θ, intensity_img, mask, width=300, height=300, overlay_color=[255, 0, 0], alpha=0.3):
    """Create a thumbnail with the correct orientation (horizontal cell, flagellum pointing to the right) and the cell
    highlighted.
    """
    # We need to cut out a bigger image if possible if we need to rotate later.
    rwidth = int(2**0.5*width+1)
    rheight = int(2**0.5*height+1)
    
    # Get the center of the bounding box
    x, y, xx, yy = bbox
    cx, cy = (xx + x) // 2, (yy + y) // 2
    
    # Constrain the width or height if we are close to the image boundary
    w = rwidth if cx > rwidth/2 else 2*cx
    h = rheight if cy > rheight/2 else 2*cy
    
    # Get the thumbnail and rotate such that the flagellum points right
    img = intensity_img[cx-w//2 : cx + w//2, cy - h//2 : cy + h//2]
    img = rotate(img, -θ + 90, resize=False)
    
    # Now cut it to the requested width/height if possible
    real_w, real_h = img.shape
    cut_w, cut_h = np.clip([real_w//2-width//2, real_h//2-height//2], 0, None)
    img = img[cut_w:-cut_w-1, cut_h:-cut_h-1]
    if (np.array(img.shape) == 0).any():
        return None

    # Do the same for the label mask
    mask = rotate(mask, -θ + 90, resize=True).astype(bool)
    padx = (img.shape[0] - mask.shape[0])/2
    pady = (img.shape[1] - mask.shape[1])/2
    mask = np.pad(mask, ((int(np.floor(padx)), int(np.ceil(padx))), (int(np.floor(pady)), int(np.ceil(pady)))))
    overlay_color = np.array(overlay_color)
    # thr = label_img[cx-w//2 : cx + w//2, cy - h//2 : cy + h//2] == r.label
    # thr = thr[:,:,np.newaxis] * np.array(overlay_color)[np.newaxis, np.newaxis]
    # thr = rotate(1.0*thr, -θ + 90, resize=True)[cut_w:-cut_w-1, cut_h:-cut_h-1]
    
    rgb = (np.array([255, 255, 255])[np.newaxis, np.newaxis]*((img-img.min())/(img.max()-img.min()))[:, :, np.newaxis]).astype(np.uint8)
    rgb[mask] = alpha*overlay_color[np.newaxis] + (1-alpha)*rgb[mask]
    return rgb

class ImageSet:
    def __init__(self, path, name=None,
                 unsharp_mask_settings={"radius": range(1, 35, 5), "amount": 0.4},
                 phase_threshold_settings={"n_standard_deviations": 1},
                 threshold_closing_settings={"selem": np.ones((5, 5))},
                 segmentation_filter_settings={"n_standard_deviations": 2, "area_bounds": (2000, 7000)},
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
    def thresholded_phase_image(self):
        mean = np.mean(self.sharpened_phase_image)
        std = np.std(self.sharpened_phase_image)
        threshold = mean-self._settings["phase_threshold"]["n_standard_deviations"]*std
        return self.sharpened_phase_image < threshold
    
    @functools.cached_property
    def closed_thresholded_phase_image(self):
        selem = self._settings["threshold_closing"]["selem"]
        return (
            dilation(
                dilation(
                    erosion(
                        erosion(
                            closing(255*self.thresholded_phase_image, selem=selem),
                            selem=selem),
                        selem=selem),
                    selem=selem),
                selem=selem))
    
    @functools.cached_property
    def labelled_threshold_image(self):
        return label(self.closed_thresholded_phase_image)
    
    @functools.cached_property
    def identified_cells(self):
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
        label_img = self.labelled_threshold_image
        intensity_img = self.phase_image

        # Calculate the minimum intensity threshold
        min_intensity_threshold = intensity_img.mean() - self._settings["segmentation_filter"]["n_standard_deviations"]*intensity_img.std()
        
        area_bounds = self._settings["segmentation_filter"]["area_bounds"]

        # Iterate through all detected separate regions in the image, filtering those we do not want to see.
        regions = []
        min_intensities = []
        for region in tqdm(regionprops(label_img, intensity_image=intensity_img)):
            forget = False
            c = np.array(region.bbox).reshape((2, 2)).T
            min_region_intensity = region.intensity_image[region.intensity_image > 0].min()
            min_intensities.append(min_region_intensity)

            # Do not consider the region if...
            if (c == 0).any() or (c == label_img.shape).any():
                # ... it touches any image border, or...
                forget = True
            elif (region.area < area_bounds[0]) or (region.area > area_bounds[1]):
                # ... it's area is too small or too large, or...
                forget = True
            elif min_region_intensity > min_intensity_threshold:
                # ... it's minimum intensity is above the mean minus (typically) two standard deviations of the full image.
                # This criterion gets rid of out of focus cells, because only in-focus cells have low enough intensity
                # somewhere in their interior in a phase contrast image. Out of focus cells do not achieve this due to
                # stray light.
                forget = True

            # Remove the region from the binary mask if we want to forget it
            if forget:
                label_img[label_img == region.label] = 0
            else:
                # ... otherwise append it to the list.
                regions.append(region)
                
        cells = SegmentedCellCollection(self.name, self.path)
        pixelsize = self.pixelsize
        for r in regions:
            # Get the center of the region's bounding box
            bbox = np.array(r.bbox).reshape(2, 2)
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

            thumb = image_thumbnail(r.bbox, 180*θ/np.pi, self.phase_image, r.image)

            cells.cell_list.append(SegmentedCell(
                length=r.major_axis_length*pixelsize,
                width=r.minor_axis_length*pixelsize,
                area=r.area*pixelsize**2,
                image=Image.fromarray(thumb)))

        self._min_intensities = min_intensities
        return cells
