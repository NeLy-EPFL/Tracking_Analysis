from scipy import signal
import numpy as np

# Compute low-pass filtered data

def butter_lowpass_filter(
    data,
    cutoff,
    order,
):
    # normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, cutoff, btype="low", analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# Compute bootstrapped confidence interval

rg = np.random.default_rng()


def draw_bs_rep(data, func, rg):
    """Compute a bootstrap replicate from data."""
    bs_sample = rg.choice(data, size=len(data))
    return func(bs_sample)


def draw_bs_ci(data, func=np.mean, rg=rg, n_reps=2000):
    """Sample bootstrap multiple times and compute confidence interval
        arguments:
            data: array-like
            func: function to compute confidence interval
            rg: random generator
            n_reps: number of bootstrap replicates
    """
    bs_reps = np.array([draw_bs_rep(data, func, rg) for _ in range(n_reps)])
    conf_int = np.percentile(bs_reps, [2.5, 97.5])
    return conf_int