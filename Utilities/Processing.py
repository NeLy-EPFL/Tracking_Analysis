from scipy import signal
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import h5py

# Compute low-pass filtered data


def butter_lowpass_filter(data, cutoff, order):
    """
    Apply a Butterworth low-pass filter to the data. This is a useful filter for removing high-frequency noise from data,
    but it can smooth out sharp edges in the data.

    Args:
        data (np.ndarray): The data to be filtered.
        cutoff (float): The cutoff frequency in normalized frequency.
        order (int): The order of the filter.

    Returns:
        y (np.ndarray): The filtered data.
    """
    # Get the filter coefficients
    b, a = signal.butter(order, cutoff, btype="low", analog=False)
    # Apply the filter to the data
    y = signal.filtfilt(b, a, data)

    return y


def savgol_lowpass_filter(data, window_length, polyorder):
    """
    Apply a Savitzky-Golay low-pass filter to the data. This is useful for removing high-frequency noise from data while
    preserving sharp edges in the data.

    Args:
        data (np.ndarray): The data to be filtered.
        window_length (int): The length of the filter window (i.e., the number of coefficients). Must be an odd integer.
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        y (np.ndarray): The filtered data.
    """

    # Apply the Savitzky-Golay filter
    y = signal.savgol_filter(data, window_length, polyorder)
    return y


def cheby1_lowpass_filter(data, cutoff, order, rp):
    """
    Apply a Chebyshev Type I low-pass filter to the data. This filter minimizes the worst-case error between the ideal
    and actual frequency response.

    Args:
        data (np.ndarray): The data to be filtered.
        cutoff (float): The cutoff frequency in normalized frequency.
        order (int): The order of the filter.
        rp (float): The maximum ripple allowed below unity gain in the passband (specified in decibels).

    Returns:
        y (np.ndarray): The filtered data.
    """

    # Get the filter coefficients
    b, a = signal.cheby1(order, rp, cutoff, btype="low", analog=False)
    # Apply the filter to the data
    y = signal.filtfilt(b, a, data)
    return y


# Compute bootstrapped confidence interval

rg = np.random.default_rng()


def draw_bs_rep(data, func, rg):
    """
    Compute a bootstrap replicate from data.

    Args:
        data (np.ndarray): The data to draw samples from.
        func (callable): The function to apply to the bootstrap samples.
        rg (np.random.Generator): The random number generator.

    Returns:
        float: The bootstrap replicate.
    """

    bs_sample = rg.choice(data, size=len(data))
    return func(bs_sample)


def draw_bs_ci(data, func=np.mean, rg=rg, n_reps=300):
    """
    Sample bootstrap multiple times and compute confidence interval.

    Args:
        data (np.ndarray): The data to draw samples from.
        func (callable, optional): The function to apply to the bootstrap samples. Defaults to np.mean.
        rg (np.random.Generator): The random number generator.
        n_reps (int, optional): The number of bootstrap replicates. Defaults to 300.

    Returns:
        np.ndarray: The lower and upper confidence interval bounds.
    """

    with ThreadPoolExecutor() as executor:
        bs_reps = list(
            tqdm(
                executor.map(lambda _: draw_bs_rep(data, func, rg), range(n_reps)),
                total=n_reps,
            )
        )
    conf_int = np.percentile(bs_reps, [2.5, 97.5])
    return conf_int


def extract_coordinates(h5_file):
    """
    Extracts the x and y coordinates from a h5 file. Only works for single object tracking. For skeleton tracking,
    use extract_skeleton.

    Args:
        h5_file (str): The path to the h5 file.

    Returns:
        tuple: Two np.ndarrays representing the x and y coordinates.
    """

    with h5py.File(h5_file, "r") as f:
        locs = f["tracks"][:].T
        y = locs[:, :, 1, :].squeeze()
        x = locs[:, :, 0, :].squeeze()
    return x, y


def replace_nans_with_previous_value(arr):
    """
    Replace NaN values with the previous value in a numpy array. If the first value is NaN, it is replaced with the
    next non-NaN value.

    Args:
        arr (np.ndarray): The array to be processed.

    Returns:
        None. The function modifies the array in-place.
    """

    # Check if the first value is NaN
    if np.isnan(arr[0]):
        # Find the next non-NaN value
        next_val = arr[next((i for i, x in enumerate(arr) if not np.isnan(x)), None)]
        arr[0] = next_val

    # Find the indices of the NaN values
    nan_indices = np.where(np.isnan(arr))

    # Replace the NaN values with the previous value
    for i in nan_indices[0]:
        arr[i] = arr[i - 1]
