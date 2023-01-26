from scipy import signal

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