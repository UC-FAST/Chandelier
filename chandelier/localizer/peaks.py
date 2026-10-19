"""Peak picking for 1-D and 2-D arrays, used by the DUET histogram.

The search is greedy: take the global maximum, record its index, then
zero a window around it so a broad peak is not reported several times.
The input is min-max scaled internally, so ``threshold`` is relative to
the observed dynamic range and the caller's array is left untouched.
"""

import numpy as np
import warnings


def find_peak_indices(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """
    This function will find the indices of the peaks of an input n-dimensional numpy array.
    This can be configured to find max or min peak indices, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max). Used exactly the same as :func:`find_peak_values`.

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use :func:`find_peak_values`.

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.

    Args:
        input_array: a 1- or 2- dimensional numpy array that will be inspected.
        n_peaks: (int) maximum number of peaks to find
        min_dist: (int) minimum distance between peaks. Default value: len(input_array) / 4
        do_min: (bool) if True, finds indices at minimum value instead of maximum
        threshold: (float) the value (scaled between 0.0 and 1.0)

    Returns:
        peak_indices: (list) list of the indices of the peak values

    """
    # Copy to float so the in-place scaling below does not mutate the
    # caller's array.
    input_array = np.array(input_array, dtype=float)

    if input_array.ndim > 2:
        raise ValueError("Cannot find peak indices on data greater than 2 dimensions!")

    is_1d = input_array.ndim == 1
    zero_dist = zero_dist0 = zero_dist1 = None
    # Default spacing separates peaks by about a quarter of the axis.
    min_dist = len(input_array) // 4 if min_dist is None else min_dist

    if is_1d:
        zero_dist = min_dist
    else:
        if type(min_dist) is int:
            zero_dist0 = zero_dist1 = min_dist
        elif len(min_dist) == 1:
            zero_dist0 = zero_dist1 = min_dist[0]
        else:
            zero_dist0, zero_dist1 = min_dist

    # scale input_array between [0.0, 1.0]
    # This makes the threshold relative to the dynamic range; constant
    # input has zero span and is intentionally left as zeros.
    input_array -= np.min(input_array)
    span = np.max(input_array)
    if span > 0:
        input_array /= span

    # flip sign if doing min
    input_array = -input_array if do_min else input_array

    # throw out everything below threshold
    # Zeroing (rather than removing) keeps the array shape for argmax.
    input_array = np.multiply(input_array, (input_array >= threshold))

    # check to make sure we didn't throw everything out
    # argmax on an all-zero array would silently return index 0.
    if np.size(np.nonzero(input_array)) == 0:
        raise ValueError("Threshold set incorrectly. No peaks above threshold.")
    if np.size(np.nonzero(input_array)) < n_peaks:
        warnings.warn("Threshold set such that there will be less peaks than n_peaks.")

    peak_indices = []
    # Extract peaks one at a time, zeroing a neighbourhood after each.
    for i in range(n_peaks):
        # np.unravel_index for 2D indices e.g., index 5 in a 3x3 array should be (1, 2)
        # Also, wrap in list for duck typing
        cur_peak_idx = list(
            np.unravel_index(np.argmax(input_array.flatten()), input_array.shape)
        )

        # zero out peak and its surroundings
        if is_1d:
            cur_peak_idx = cur_peak_idx[0]
            peak_indices.append(cur_peak_idx)
            lower, upper = _set_array_zero_indices(
                cur_peak_idx, zero_dist, len(input_array)
            )
            input_array[lower:upper] = 0
        else:
            peak_indices.append(cur_peak_idx)
            lower0, upper0 = _set_array_zero_indices(
                cur_peak_idx[0], zero_dist0, input_array.shape[0]
            )
            lower1, upper1 = _set_array_zero_indices(
                cur_peak_idx[1], zero_dist1, input_array.shape[1]
            )
            input_array[lower0:upper0, lower1:upper1] = 0

        # Stop once every candidate has been zeroed: argmax would keep
        # reporting the same empty index.
        if np.sum(input_array) == 0.0:
            break

    return peak_indices


def _set_array_zero_indices(index, zero_distance, max_len):
    """Return a clamped slice covering ``index ± zero_distance``."""

    lower = index - zero_distance
    # +1 makes the upper bound inclusive, since slices are half-open.
    upper = index + zero_distance + 1
    lower = 0 if lower < 0 else lower
    upper = max_len if upper >= max_len else upper
    return int(lower), int(upper)
