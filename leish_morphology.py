from pathlib import Path
import tifffile
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import json
from PIL import Image

from skimage.filters import unsharp_mask, frangi
from skimage.morphology import closing, erosion, dilation, binary_closing, binary_erosion, binary_dilation, diameter_closing, medial_axis
from skimage.measure import label, regionprops, EllipseModel
from skimage.color import label2rgb
from skimage.transform import rotate

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

def threshold_phase(img, nstd=1):
    """Threshold a phase contrast image `img` by calculating its mean and standard deviation, then applying a threshold of `mean-nstd*std`. Per default, `nstd=1`.
    """
    mean = np.mean(img)
    std = np.std(img)
    threshold = mean-nstd*std
    return img < threshold

def heal_thresholded_img(img, selem=np.ones((5, 5))):
    """Apply morphological operations to heal small holes in thresholded images, and erode away small protrusions. Optionally can pass a custom selem to run the morphological operations with.
    """
    #closed = binary_closing(thresh_img, selem=np.ones((5, 5)))
    #closed = binary_dilation(binary_dilation(binary_erosion(binary_erosion(diameter_closing(binary_erosion(binary_erosion(binary_dilation(binary_dilation(thresh_img)))), diameter_threshold=10)))))
    #closed = binary_dilation(binary_dilation(binary_erosion(binary_erosion(binary_closing(thresh_img)))))
    return dilation(dilation(erosion(erosion(closing(255*img, selem=selem), selem=selem), selem=selem), selem=selem), selem=selem)

def segment_and_filter(bin_img, intensity_img, area_bounds=(2000, 7000), intensity_stds=2, return_min_intensities=False):
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
    label_img = label(bin_img)

    # Calculate the minimum intensity threshold
    min_intensity_threshold = intensity_img.mean() - intensity_stds*intensity_img.std()

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
    
    if return_min_intensities:
        return regions, min_intensities
    return regions

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

def extract_morphology(regions, phase_img, gene_img, dapi_img, pixelsize):
    """Iterate through `regions`, extracting cell length and widths (via an ellipsoidal model), cell area and a cell thumbnail.
    Returns a `DataFrame` containing this data.
    """
    morphology = []
    for r in regions:
        # Get the center of the region's bounding box
        bbox = np.array(r.bbox).reshape(2, 2)
        c = bbox[0] + np.diff(bbox, axis=0)//2
        # Regionprops does fit an ellipse for us and gives us the major and minor axes, as well as the orientation angle.
        θ = r.orientation
        # Find the maximum of the DAPI intensity in the region. This will be the location of the kinetochore.
        kinetochore_pos = np.unravel_index((dapi_img[r.slice]*r.image).argmax(), r.image.shape) + bbox[0]
        # The orientation of the kinetochore relative to the center tells us the right/left flip of the cell
        ϕ = np.arctan2(*(kinetochore_pos - c)[0][::-1])
        # Adjust the orientation angle such that if we later rotate the image, the flagellum will point right.
        if np.cos(θ)*np.cos(ϕ)+np.sin(θ)*np.sin(ϕ) < 0:
            θ += np.pi
            
        thumb = image_thumbnail(r.bbox, 180*θ/np.pi, phase_img, r.image)
        
        morphology.append({
            "cell_length": r.major_axis_length*pixelsize,
            "cell_width": r.minor_axis_length*pixelsize,
            "area": r.area*pixelsize**2,
            "orientation_angle": 180*θ/np.pi,
            "thumbnail": Image.fromarray(thumb) if thumb is not None else None,
        })
    return pd.DataFrame(morphology)

def process_image(path):
    """Process the micromanager image directory given in `path`. This expects three channel files, phase contrast, gene image
    and DAPI image in this order. It extracts the pixel size from the phase contrast image.
    """
    path = Path(path)
    # Get image paths
    phase_path, gene_path, dapi_path = sorted([x for x in path.glob('img_channel???_position000_time000000000_z000.tif')])
    # Open phase contrast and image metadata
    with tifffile.TiffFile(phase_path, 'rb') as tif:
        phase = tif.asarray()
        metadata = json.loads(tif.imagej_metadata["Info"])
    # Open gene image (not used yet)
    with tifffile.TiffFile(gene_path, 'rb') as tif:
        gene = tif.asarray()
    # Open DAPI image
    with tifffile.TiffFile(dapi_path, 'rb') as tif:
        dapi = tif.asarray()
    # Get pixel size
    pixelsize = metadata["PixelSizeUm"]
    
    # Sharpen
    uns = unsharp_mask_multi(phase, range(1, 35, 5), 0.4)
    # Threshold and heal
    thresh_img = threshold_phase(uns)
    closed = heal_thresholded_img(thresh_img)
    # Process regions
    regions = segment_and_filter(closed, phase)
    # Extract morphology
    df = extract_morphology(regions, phase, gene, dapi, pixelsize)
    
    return df