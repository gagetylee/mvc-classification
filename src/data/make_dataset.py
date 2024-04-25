## LOAD SIGNALS
import os
import pandas as pd
import numpy as np

task_names = {
  'e': 'extension',
  'f': 'flexion',
  'p': 'pronation',
  's': 'supination'
}
sfreq = 2048
num_channels = 120
s_duration = 10

def load_subject_attributes(path):
    df = pd.read_csv(path, sep="\t", header=0)
    df['subject'] = df['subject'].astype('string')
    return df

def load_metadata(path):
    columns = ['subject', 'task', '%mvc', 'bicep_path', 'torque_path']
    rows = []
    curr_row = 0
    for dirpath, dirnames, files in os.walk(path):
        if dirpath.endswith('biceps'):
            for filename in files:
                if filename.endswith('.bin'):
                    basename = os.path.splitext(filename)[0]
                    bb_sample_path = os.path.join(dirpath, filename)
                    details = basename.split('_')

                    # Get corresponding torque data
                    torque_path = bb_sample_path.replace('biceps', 'torque')
                    torque_path = torque_path.replace('bb', 'torque')

                    row = {
                    'subject': details[0],
                    'task': task_names[details[1][0]],
                    '%mvc': int(details[1][1:]),
                    'bicep_path': bb_sample_path,
                    'torque_path': torque_path
                    }
                    rows.append(row)
                    curr_row += 1
    meta = pd.DataFrame(rows, columns=columns).sort_values('subject', ignore_index=True)
    meta.reset_index()
    meta['subject'] = meta['subject'].astype('string')

    # Join with subject attributes
    attributes_path = os.path.join(path, 'SubjectsDescription.txt')
    details = load_subject_attributes(attributes_path)
    
    meta = meta.merge(details, on='subject', how='left')
    return meta


def load_signals(metadata, nrows=10):
    emg_list = []
    dtype = np.dtype('<f8')
    
    for signal_no, row in metadata.head(nrows).iterrows():
        data = np.fromfile(row['bicep_path'], dtype=dtype)
        reshaped_data = data.reshape((-1, num_channels))
        num_samples = reshaped_data.shape[0]

        sample_indices = np.arange(num_samples)
        
        index = pd.MultiIndex.from_arrays([[signal_no] * num_samples, sample_indices], names=['signal', 'sample'])
        temp_df = pd.DataFrame(reshaped_data, index=index)
        emg_list.append(temp_df)
        
    df = pd.concat(emg_list)
    return df