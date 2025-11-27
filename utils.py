import numpy as np
import torch
import multiprocessing as mp
from pqdm.processes import pqdm
from Segment.UpdatedAutomatedPWPID import segmentation_utils as seg
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
        (thresholds[0], thresholds[1] - thresholds[0], hist.numel() - thresholds[1],),
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
    return np.array(list(A))

def int2float(array):

    return ((array/255)*2)-1

def threshold_sweep_inline(grads,b_thresh,t_thresh,n_thresh):
    '''
    This function is the main wrapper for the gradient threshold step 
    of watershed segmentation.
    '''
    
    # Performs basic checks on inputs
    seg.check_threshs(b_thresh,t_thresh,n_thresh)    
    # Creates an array of thresholds to test
    thresholds = np.linspace(b_thresh,t_thresh,int(n_thresh))
    # Prepares inputs for parallel operation
    inputs = [(grads,threshold) for threshold in thresholds]
    # Gets an array of the number of markers for each threshold
    num_markers = pqdm(inputs,seg.gradient_threshold,n_jobs=32,disable=True)
    
    return thresholds,num_markers

def max_marker_grad_thresh(threshs,num_markers):
    '''
    Plots and saves the results from the gradient threshold sweep
    Outputs the threshold that yielded the maximum number of markers. 
    '''

    max_marker_thresh = threshs[np.argmax(num_markers)]

    return max_marker_thresh

def watershed_volume(img,grad,thresh):
    
    #print("Thresholding gradient image...")

    thresh_grad = seg.threshold_grad(grad,thresh)
            
    thresh_grad = np.uint8(thresh_grad)

    # print("Done.")
    # print("Performing connected component analysis to identify markers...")

    markers = cc3d.connected_components(thresh_grad,connectivity=6)
            
    # print("Done.")
    # print("Applying watershed transform...")
    seg_img = watershed(grad,markers,watershed_line=False)
    # print("Done.")

    # print("Analyzing distribution of marker greyscales...")

    avg_grey, marker_size, size = seg.analyze_seg_dist_markers(img,seg_img,markers)

    avg_img = seg.plot_avg_gray(seg_img,avg_grey)

    # print("Done.")
    
    return avg_grey, marker_size,size,seg_img,avg_img

def analyze_post_watershed_dist(avg_grey,size):

    counts, bin_edges = np.histogram(avg_grey, weights=size, density=True,bins=np.linspace(np.amin(avg_grey),np.amax(avg_grey),255))
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

    return min1_x,min2_x

def segment(array:np.array,thresh_min=None,thresh_max=None,n_thresh=50,thresh=0.017,bg_thresh=50,gw_thresh=138):
    """
    Array should be of the shape (N,H,W,D)
    """
    seg_vols = []
    for i,subvol in enumerate(array):
        try:
            grads = seg.sobel_gradients(subvol)
            # Run gradient threshold sweep
            if thresh is None:
                thresholds, num_markers = threshold_sweep_inline(grads,thresh_min,thresh_max,n_thresh)
            # Plot threshold swep results and find maximum marker threshold. 
                max_marker_thresh = max_marker_grad_thresh(thresholds,num_markers)

                #print("Maximum marker threshold:",round(max_marker_thresh,3))

            max_marker_thresh = thresh if thresh is not None else max_marker_thresh
            print("Maximum marker threshold:",round(max_marker_thresh,3))
            # Watershed image
            avg_grey, marker_size,size,seg_img,avg_img = watershed_volume(subvol,grads,max_marker_thresh)

            # Find minima
            if (bg_thresh is None) and (gw_thresh is None):
                bg_thresh, gw_thresh = analyze_post_watershed_dist(avg_grey,size)
            else:
                bg_thresh,gw_thresh = bg_thresh,gw_thresh

            assert(avg_img.shape==subvol.shape), "avg_img and subvol should be the same shape"
            # Phase-ID image
            
            print(avg_img.flatten()[:6])

            bg_thresh = ((bg_thresh/255)*2)-1 if type(bg_thresh) is int else bg_thresh
            gw_thresh = ((gw_thresh/255)*2)-1 if type(gw_thresh) is int else gw_thresh

            #plt.hist(avg_img.ravel(), bins=100, color='blue', alpha=0.7)
            #plt.axvline(bg_thresh, color='red', linestyle='dashed', linewidth=1, label='Black-Grey Threshold')
            #plt.axvline(gw_thresh, color='green', linestyle='dashed', linewidth=1, label='Grey-White Threshold')
            #plt.show()

            seg_img = seg.threshold_volume(avg_img,bg_thresh,gw_thresh)
            seg_vols.append(seg_img)
        except Exception as e:
            print(f"Error processing volume {i}: {e}")
            seg_vols.append(np.zeros_like(subvol))
    seg_vols = np.array(seg_vols)
    return seg_vols
