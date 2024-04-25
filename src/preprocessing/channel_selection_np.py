import numpy as np
import pandas as pd
from scipy.signal import welch

def calculate_psd(data, fs=2048):
    """Calculate the Power Spectral Density for each channel.
    
    Parameters:
    - df: {array-like} (sample, data)
    - fs: Sampling frequency.
    
    Returns:
    - A list of PSD arrays for each channel.
    """
    psds = []
    for i in range(data.shape[1]):
        _, psd = welch(data[:, i], fs=fs)
        psds.append(psd)
    return psds

def normalize_psd(psds):
    """Normalize PSD values against the maximum PSD over all channels.
    
    Parameters:
    - psds: List of PSD arrays for each channel.
    
    Returns:
    - Normalized PSD values for each channel.
    """
    max_psd = np.max([np.max(psd) for psd in psds])
    normalized_psds = [psd / max_psd for psd in psds]
    return normalized_psds

def pearson_correlation(data):
    """Calculate Pearson's correlation coefficient matrix from DataFrame.
    
    Parameters:
    - data: {array-like} (sample, data)
    
    Returns:
    - Pearson's correlation coefficient matrix.
    """
    return np.corrcoef(data, rowvar=False)

def mean_absolute_correlation(corr_matrix):
    """Calculate mean absolute correlation for each channel.
    
    Parameters:
    - corr_matrix: Correlation coefficient matrix.
    
    Returns:
    - Mean absolute correlation for each channel.
    """
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 0)  # Ignore self-correlation
    mean_corr = abs_corr.mean(axis=1)
    return mean_corr

def calculate_pcr(normalized_psds, mean_corr):
    """Calculate PCR for each channel.
    
    Parameters:
    - normalized_psds: Normalized PSDs for each channel.
    - mean_corr: Mean absolute correlation for each channel.
    
    Returns:
    - PCR for each channel.
    """
    pcr = [np.mean(psd) / mc for psd, mc in zip(normalized_psds, mean_corr)]
    return pcr

def extract_channels(df, num_channels):
    """Extract the most relevent channels using the Power-Correlation Ratio Maximization method.
    
    Parameters:
    - df: {DataFrame} (sample, data)
        time-series signal data
    - num_channels: number of channels to extract

    Returns:
    - Extracted channel data
    """
    data = df.copy().to_numpy()

    psds = calculate_psd(data)
    normalize_psds = normalize_psd(psds)
    corr_matrix = pearson_correlation(data)
    mean_corr = mean_absolute_correlation(corr_matrix)
    pcr = calculate_pcr(normalize_psds, mean_corr)

    selected_channels = np.argsort(pcr)[:num_channels].tolist()
    emg_df = df.loc[:][selected_channels]

    return emg_df

