import pandas as pd
import numpy as np

def time_features(df):
    signals = df.index.get_level_values(0).unique().tolist()
    channels = df.columns.tolist()
    features = ['max', 'std', 'mean']
    
    new_index = pd.MultiIndex.from_product([signals, channels], names=['signal', 'channel'])
    feature_table = pd.DataFrame(index=new_index, columns=features)
    
    for signal in signals:
        signal_data = df.loc[signal]
        for channel in channels:
            channel_data = signal_data[channel]
            
            feature_table.loc[(signal, channel), 'max'] = channel_data.max()
            feature_table.loc[(signal, channel), 'std'] = channel_data.std()
            feature_table.loc[(signal, channel), 'mean'] = channel_data.mean()

    return feature_table

def mean_frequency(signal, sfreq):
    fft_vals = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(len(signal), 1/sfreq)
    fft_power = np.abs(fft_vals)**2
    
    mean_freq = np.sum(fft_freq * fft_power) / np.sum(fft_power)
    return mean_freq

def first_pca_coeff(signal):
    pca = PCA(n_components=1)
    pca.fit(signal)
    coefficient = pca.components_[0,0]
    return coefficient

def frequency_features(df, sfreq):
    signals = df.index.get_level_values(0).unique().tolist()
    channels = df.columns.tolist()
    features = ['mean_freq']
    
    new_index = pd.MultiIndex.from_product([signals, channels], names=['signal', 'channel'])
    feature_table = pd.DataFrame(index=new_index, columns=features)

    for signal in signals:
            signal_data = df.loc[signal]
            for channel in channels:
                channel_data = signal_data[channel]
                
                feature_table.loc[(signal, channel), 'mean_freq'] = mean_frequency(channel_data.values.flatten(), sfreq)
               
    return feature_table

def subject_metadata(df, metadata):
    copy = df.copy()
    copy = copy.reset_index(level=['signal', 'channel'])
    copy = copy.join(df)

def feature_table(df, subject_data, sfreq):
    copy = df.copy()

    time_features_df = time_features(copy)
    freq_features_df = frequency_features(copy, sfreq)

    feature_table = time_features_df.merge(freq_features_df, on=['signal', 'channel'])
    feature_table = feature_table.join(subject_data[['age', 'height', 'weight_kg', '%mvc']], on='signal')
    
    return feature_table.reset_index(level=['signal', 'channel'])
    


