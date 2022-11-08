from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

from .segmentation import ImageSet

def _analyse_image_path(p):
    return ImageSet(p[0], p[1]).identified_cells

def _analyse_image_path_collection(paths_names):
    results = []
    with tqdm(total=len(paths_names), smoothing=0) as pbar:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_analyse_image_path, p) for p in paths_names]
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update()
    return results
