#### Load package ####

import itertools
from PIL import Image
import pickle
import os
import PIL
import numpy as np
import pandas as pd
import yaml
import tifffile as tiff
import glob
from time import time
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tracemalloc
import tempfile, shutil
from functools import wraps
import cv2
from numba import njit, prange
import gc
#import pyvips
#import matplotlib as plt

Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = 10e100


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


#def load_image(filename, verbose=True):
#    img = Image.open(filename)
#    img = np.array(img)
#    if img.ndim == 3 and img.shape[-1] == 4:
#        img = img[..., :3]  # remove alpha channel
#    if verbose:
#        print(f'Image loaded from {filename}')
#    return img

# modified raw load_image


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask



def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)


def load_yaml(filename, verbose=False):
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]

def prepare_for_clustering(embs, location_weight):
    mask = np.all([np.isfinite(c) for c in embs], axis=0)
    embs = np.stack([c[mask] for c in embs], axis=-1)

    if location_weight is None:
        x = embs
    else:
        embs -= embs.mean(0)
        embs /= embs.var(0).sum()**0.5
        # get spatial coordinates
        locs = np.meshgrid(
                *[np.arange(mask.shape[i]) for i in range(mask.ndim)],
                indexing='ij')
        locs = np.stack(locs, -1).astype(float)
        locs = locs[mask]
        locs -= locs.mean(0)
        locs /= locs.var(0).sum()**0.5

        # balance embeddings and coordinates
        embs *= 1 - location_weight
        locs *= location_weight
        x = np.concatenate([embs, locs], axis=-1)
    return x, mask

def cluster_mask(
        embs, n_clusters, method='mbkm', location_weight=None,
        sort=True):

    x, mask = prepare_for_clustering(embs, location_weight)

    print(f'Clustering pixels using {method}...')
    t0 = time()
    if method == 'mbkm':
        model = MiniBatchKMeans(
                n_clusters=n_clusters,
                # batch_size=x.shape[0]//10, max_iter=1000,
                # max_no_improvement=100, n_init=10,
                random_state=0, verbose=0)
    elif method == 'km':
        model = KMeans(
                n_clusters=n_clusters,
                random_state=0, verbose=0)
    elif method == 'gm':
        model = GaussianMixture(
                n_components=n_clusters,
                covariance_type='diag', init_params='k-means++',
                random_state=0, verbose=1)
    # elif method == 'dbscan':
    #     eps = x.var(0).sum()**0.5 * 0.5
    #     min_samples = 5
    #     model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=64)
    elif method == 'hdbscan':
        min_cluster_size = min(1000, x.shape[0] // 400 + 1)
        min_samples = min_cluster_size // 10 + 1
        model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                core_dist_n_jobs=64)
    elif method == 'agglomerative':
        # knn_graph = kneighbors_graph(x, n_neighbors=10, include_self=False)
        model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward', compute_distances=True)
    else:
        raise ValueError(f'Method `{method}` not recognized')
    print(x.shape)
    labels = model.fit_predict(x)
    print(int(time() - t0), 'sec')
    print('n_clusters:', np.unique(labels).size)

    if sort:
        labels, order = sort_labels(labels)

    labels_arr = np.full(mask.shape, labels.min()-1, dtype=int)
    labels_arr[mask] = labels

    # if method == 'gm':
    #     probs = model.predict_proba(embs)
    #     probs = probs[:, order]
    #     assert (probs.argmax(-1) == labels).all()
    #     probs_arr = np.full(
    #             mask.shape+(n_clusters,), np.nan, dtype=np.float32)
    #     probs_arr[mask] = probs
    # else:
    #     probs_arr = None

    return labels_arr, model



    #### here







def get_peak_rss():
    """Return the peak resident set size in GB"""
    return _PEAK_RSS_GB



def measure_peak_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        current_gb = current / (1024 ** 3)
        peak_gb = peak / (1024 ** 3)
        print(f"[{func.__name__}] Current memory: {current_gb:.4f} GB; Peak memory: {peak_gb:.4f} GB")
        tracemalloc.stop()
        return result
    return wrapper


#### Basic functions ####

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

def find_he(base_dir, stem):
    exts = ['.tiff', '.tif', '.svs', '.ome.tif',  '.ome.tiff',  '.jpg', '.png','.ndpi', '.scn', '.mrxs']
    for ext in exts:
        candidate = os.path.join(base_dir, f"{stem}{ext}")
        #print(candidate)
        if os.path.exists(candidate):
            return candidate
    return None

def patchify_memmap(image, patch_size, tmp_dir=None):
    """
    Patchify large image using a disk-backed memmap.
    Returns (memmap_file_path, patches_memmap, shapes)
    """
    h, w, c = image.shape
    tiles_shape = (h // patch_size, w // patch_size)

    # make temporary directory
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="histosweep_memmap_")
    os.makedirs(tmp_dir, exist_ok=True)

    # create memmap file on disk
    mmap_path = os.path.join(tmp_dir, "patches.dat")
    patches = np.memmap(mmap_path, dtype=np.uint8, mode="w+",
                        shape=(tiles_shape[0] * tiles_shape[1],
                               patch_size, patch_size, c))

    # fill patches in blocks to avoid memory blowup
    idx = 0
    for i in range(tiles_shape[0]):
        for j in range(tiles_shape[1]):
            patch = image[i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size, :]
            patches[idx] = patch
            idx += 1
    patches.flush()

    shapes = dict(tiles=np.array(tiles_shape), tmp_dir=tmp_dir, mmap_path=mmap_path)
    return mmap_path, patches, shapes


def cleanup_memmap(cache_root):
    """Remove the temporary memmap directory and all files inside."""
    try:
        if cache_root and os.path.isdir(cache_root):
            shutil.rmtree(cache_root)
            print(f"🧹 Cleaned up memmap cache: {cache_root}")
    except Exception as e:
        print(f"⚠️ Failed to cleanup memmap cache {cache_root}: {e}")


def get_image_filename(prefix, printName = True):
    print("\nLooking for files with prefix:", prefix)
    #print("Current working directory:", os.getcwd())
    print("All matching files:", glob.glob(prefix + ".*"))
    
    for suffix in ['.jpg', '.png', '.tiff', '.tif', '.svs', '.ome.tif', '.ndpi']:
        filename = prefix + suffix
        if printName:
            print(f"Checking: {filename}")
        if os.path.exists(filename):
            print(f"✅ Found: {filename}")
            return filename
    raise FileNotFoundError(f'Image not found with prefix: {prefix}')



def smart_save_image(img, prefix, base_name="base", size_threshold=5000):
    """
    Save image as JPG if both dimensions are under `size_threshold`, otherwise as TIFF.
    """
    h, w = img.shape[:2]
    print(f"Image size: {h}x{w}")

    if h < size_threshold and w < size_threshold:
        # Save as JPG
        path = f"{prefix}{base_name}.jpg"
        Image.fromarray(img.astype(np.uint8)).save(path, quality=100)
        print(f"✅ Saved as JPG: {path}")
    else:
        # Save as TIFF
        path = f"{prefix}{base_name}.tiff"
        tifffile.imwrite(path, img, bigtiff=True)
        print(f"✅ Saved as TIFF: {path}")

def smart_save_image_vips(img, prefix, base_name="base", size_threshold=5000):
    """
    Save image as JPG if both dimensions are under `size_threshold`, otherwise as TIFF.
    Accepts NumPy array or pyvips.Image.
    """
    import pyvips
    vips_img = to_vips(img)
    h, w = vips_img.height, vips_img.width
    print(f"Image size: {h}x{w}")

    if h < size_threshold and w < size_threshold:
        path = f"{prefix}{base_name}.jpg"
        vips_img.jpegsave(path, Q=100)
        print(f"✅ Saved as JPG: {path}")
    else:
        path = f"{prefix}{base_name}.tiff"
        vips_img.tiffsave(path, bigtiff=True, compression="lzw", tile=True, pyramid=True)
        print(f"✅ Saved as TIFF: {path}")


# modified raw load_image
def load_image(filename, verbose=True):
    ext = os.path.splitext(filename)[-1].lower()
    print(f"\nfilename = {filename},ext={ext}")
    # Whole-slide formats: SVS, NDPI, OME-TIFF, SCN, MRXS etc.
    if ext in [".svs", ".ndpi", ".scn", ".mrxs", ".ome.tif", ".ome.tiff"]:
        print("This is a whole-slide format (svs/ndpi/ome/scn/mrxs)... using pyvips")
        import pyvips
        slide = pyvips.Image.new_from_file(filename, access="sequential")
        # Get image properties
        print(f"Width: {slide.width}, Height: {slide.height}, Bands: {slide.bands}")
        # Convert to NumPy array
        img = np.ndarray(
            buffer=slide.write_to_memory(),
            dtype=np.uint8,
            shape=(slide.height, slide.width, slide.bands)
        )
        print(img.shape)
    if ext in ['.tif', '.tiff']:
        print("this is tif|tiff file......")
        with tifffile.TiffFile(filename) as tif:
            img = tif.pages[0].asarray()  # always just take the first page
    if ext in ['.jpg', '.jpeg', '.png']:
        print("this is jpg|jpeg|png file......")
        img = Image.open(filename)
        img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def to_vips(img):
    """Convert NumPy or PyVIPS image to PyVIPS.Image."""
    import pyvips
    if isinstance(img, pyvips.Image):
        return img
    elif isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:
            return pyvips.Image.new_from_memory(img.tobytes(), w, h, 1, 'uchar')
        elif img.ndim == 3 and img.shape[2] == 3:
            return pyvips.Image.new_from_memory(img.tobytes(), w, h, 3, 'uchar')
        else:
            raise ValueError("Unsupported NumPy array shape")
    else:
        raise TypeError("Input must be a NumPy array or pyvips.Image")


def load_image_vips(filename):
    print("read image with pyvips.....")
    import pyvips
    image = pyvips.Image.new_from_file(filename, access='random')
    print(f"Image Width: {image.width}, Image Height: {image.height}, Image Bands: {image.bands}")
    return image

