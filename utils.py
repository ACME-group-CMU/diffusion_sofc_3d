import numpy as np
import torch
import multiprocessing as mp
from pqdm.processes import pqdm
import cc3d
import Automated_PWPID.segmentation_utils as seg
from skimage.segmentation import watershed
from scipy.signal import find_peaks, argrelmin


def vfs(array: np.array):
    """
    Array should be of the shape (N,H,W,D)
    """

    seg_vols, correct_seg = segment(array)
    vfs = get_vf_array(seg_vols)

    return vfs, correct_seg


def vf_otsu(microstructures):

    np_micro = microstructures.data.cpu().numpy()
    np_micro = (((np_micro + 1) / 2) * 255).astype(int)
    try:
        thresholds = threshold_multiotsu(np_micro)
    except:
        thresholds = [50, 130]

    # print(thresholds)
    torch.use_deterministic_algorithms(False)
    hist = (
        torch.histc(microstructures, bins=256, min=-1, max=1) / microstructures.numel()
    )
    torch.use_deterministic_algorithms(True)
    percs = torch.empty((3,))

    areas = torch.split(
        hist,
        (
            thresholds[0],
            thresholds[1] - thresholds[0],
            hist.numel() - thresholds[1],
        ),
    )

    for i in range(3):
        percs[i] = torch.sum(areas[i])

    return percs


def get_vfs(volume):
    """
    Get volume fractions from a segmented array.

    Parameters:
    -----------
    array : numpy.ndarray
        Segmented array with labels 1, 2, or 3

    Returns:
    --------
    numpy.ndarray
        Volume fractions for each label
    """
    _, counts = np.unique(volume, return_counts=True)
    vfs = counts / np.sum(counts)
    return vfs


def get_vf_array(array):
    """
    Get volume fractions for each slice in a 3D array.

    Parameters:
    -----------
    array : numpy.ndarray
        Segmented array with shape (H,W,D) or (N,H,W,D)

    Returns:
    --------
    numpy.ndarray
        Volume fractions for each slice
    """

    A = map(get_vfs, array)
    A = list(A)
    
    return np.array(list(A))


def threshold_sweep_inline(grads, b_thresh, t_thresh, n_thresh):
    """
    This function is the main wrapper for the gradient threshold step
    of watershed segmentation.
    """

    # Performs basic checks on inputs
    seg.check_threshs(b_thresh, t_thresh, n_thresh)
    # Creates an array of thresholds to test
    thresholds = np.linspace(b_thresh, t_thresh, int(n_thresh))
    # Prepares inputs for parallel operation
    inputs = [(grads, threshold) for threshold in thresholds]
    # Gets an array of the number of markers for each threshold
    num_markers = pqdm(
        inputs, seg.gradient_threshold, n_jobs=mp.cpu_count(), disable=True
    )

    return thresholds, num_markers


def max_marker_grad_thresh(threshs, num_markers):
    """
    Plots and saves the results from the gradient threshold sweep
    Outputs the threshold that yielded the maximum number of markers.
    """

    max_marker_thresh = threshs[np.argmax(num_markers)]

    return max_marker_thresh


def watershed_volume(img, grad, thresh):

    # print("Thresholding gradient image...")

    thresh_grad = seg.threshold_grad(grad, thresh)

    thresh_grad = np.uint8(thresh_grad)

    # print("Done.")
    # print("Performing connected component analysis to identify markers...")

    markers = cc3d.connected_components(thresh_grad, connectivity=6)

    # print("Done.")
    # print("Applying watershed transform...")
    seg_img = watershed(grad, markers, watershed_line=False)
    # print("Done.")

    # print("Analyzing distribution of marker greyscales...")

    avg_grey, marker_size, size = seg.analyze_seg_dist_markers(img, seg_img, markers)

    avg_img = seg.plot_avg_gray(seg_img, avg_grey)

    # print("Done.")

    return avg_grey, marker_size, size, seg_img, avg_img


def analyze_post_watershed_dist(avg_grey, size):

    counts, bin_edges = np.histogram(
        avg_grey,
        weights=size,
        density=True,
        bins=np.linspace(np.amin(avg_grey), np.amax(avg_grey), 255),
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Step 1: Find peaks (local maxima)
    peaks, _ = find_peaks(counts, distance=10)
    peak_xs = bin_centers[peaks]

    # Step 2: Find local minima
    minima_indices = argrelmin(counts)[0]
    minima_xs = bin_centers[minima_indices]

    # Step 3: Keep only the minima *between* the top 3 peaks
    top3_peaks = peaks[np.argsort(counts[peaks])[-3:]]
    top3_peaks.sort()

    minima_between_peaks = []
    for i in range(len(top3_peaks) - 1):
        left = top3_peaks[i]
        right = top3_peaks[i + 1]
        between = minima_indices[(minima_indices > left) & (minima_indices < right)]
        if len(between) > 0:
            min_idx = between[np.argmin(counts[between])]
            minima_between_peaks.append(min_idx)

    # Get x-values
    min1_x = bin_centers[minima_between_peaks[0]]
    min2_x = bin_centers[minima_between_peaks[1]]

    return min1_x, min2_x


def segment(array: np.array):

    """
    Array should be of the shape (N,H,W,D)
    """

    seg_vols = []
    seg_correct = np.ones(array.shape[0], dtype=bool)

    for num, subvol in enumerate(array):

        try:
            grads = seg.sobel_gradients(subvol)
            # Run gradient threshold sweep
            thresholds, num_markers = threshold_sweep_inline(grads, 0.0, 1.0, 100)

            # Plot threshold swep results and find maximum marker threshold.
            max_marker_thresh = max_marker_grad_thresh(thresholds, num_markers)

            # Watershed image
            avg_grey, marker_size, size, seg_img, avg_img = watershed_volume(
                subvol, grads, max_marker_thresh
            )

            # Find minima
            bg_thresh, gw_thresh = analyze_post_watershed_dist(avg_grey, size)

            assert (
                avg_img.shape == subvol.shape
            ), "avg_img and subvol should be the same shape"
            # Phase-ID image
            seg_img = seg.threshold_volume(avg_img, bg_thresh, gw_thresh)
            assert np.unique(seg_img).size == 3, 'Segmentation should yield 3 phases'
        except Exception as e:
            seg_img = np.random.randint(1,3, subvol.shape, dtype=np.uint8)
            seg_correct[num] = 0

        seg_vols.append(seg_img)
        

    seg_vols = np.array(seg_vols)
    return seg_vols, seg_correct
