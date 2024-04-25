from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(df, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return df.apply(lambda x: filtfilt(b, a, x), axis=0)

def rectify_and_smooth(df, window_size=100):
    rectified_df = np.abs(df)

    window = np.ones(window_size) / window_size
    smoothed_df = rectified_df.apply(lambda x: np.convolve(x, window, 'same'), axis=0)
    return smoothed_df

def preprocess_signal(df, fs):
    lowcut = 10.0
    highcut = 500.0
    order = 4

    copy = df.copy()

    # Band-pass filter
    processed_df = butter_bandpass_filter(copy, lowcut, highcut, fs, order)

    # Rectification
    processed_df = rectify_and_smooth(processed_df)
    
    return processed_df