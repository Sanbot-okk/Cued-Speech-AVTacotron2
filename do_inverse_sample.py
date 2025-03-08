import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import math

def filter_and_downsample(data, cutoff, fs_original, fs_new, order=5):
    nyq = 0.5 * fs_original
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    # Decimate data to downsample
    # Calculate decimation factor
    decimation_factor = math.ceil(fs_original / fs_new)
    return filtered_data[::decimation_factor]


def fit_pca(data, scaler_check):
    if scaler_check:
        scaler = StandardScaler()
        pca = PCA(n_components=10) #number of PCA components (change)
        scaled_data = scaler.fit_transform(data)
        pca.fit(scaled_data)
        return pca, scaler
    else:
        pca = PCA(n_components=10)
        pca.fit(data)
        return pca

def process_and_transform(syn_folder, lips_pca, hands_pca, lips_scaler, hands_scaler, output_folder, cutoff_frequency, fs_original, fs_new):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(syn_folder):
        if filename.endswith("_syn.npy"):
            filepath = os.path.join(syn_folder, filename)
            data = np.load(filepath)

            # Split into lips and hands
            lips = data[:, :10]
            hands = data[:, 10:20]
            #pos = data[:,20:22]

            # Downsample lips and hands
            lips_downsampled = filter_and_downsample(lips, cutoff_frequency, fs_original, fs_new)
            hands_downsampled = filter_and_downsample(hands, cutoff_frequency, fs_original, fs_new)
            #pos_downsampled = filter_and_downsample(pos, cutoff_frequency, fs_original, fs_new)

            # Transform using PCA
            lips_transformed = lips_scaler.inverse_transform(lips_pca.inverse_transform(lips_downsampled))
            hands_transformed = hands_scaler.inverse_transform(hands_pca.inverse_transform(hands_downsampled))

            # Save the transformed data
            np.save(os.path.join(output_folder, filename.replace("_syn.npy", "_lips.npy")), lips_transformed)
            np.save(os.path.join(output_folder, filename.replace("_syn.npy", "_hands.npy")), hands_transformed)
            #np.save(os.path.join(output_folder, filename.replace("_syn.npy", "_pos.npy")), pos_downsampled)

# Paths to the folders
syn_folder = "outdir/inference/synthesis/"
lips_folder = "/AVTacotron2_data/CSF23/MP_npy/MP_lips/"
hands_folder = "/AVTacotron2_data/CSF23/MP_npy/MP_hands/"
output_folder = "outdir/inference/filtered_resampled_inverted_both/"

# Parameters
fs_original = 86.7  # original sampling rate
fs_new = 30         # new sampling rate
cutoff_frequency = 15  # cutoff frequency to satisfy Nyquist

# Load and downsample data for PCA fitting
lips_data_for_pca = filter_and_downsample(np.vstack([np.load(os.path.join(lips_folder, f)) for f in os.listdir(lips_folder) if f.endswith(".npy")]), cutoff_frequency, fs_original, fs_new)
hands_data_for_pca = filter_and_downsample(np.vstack([np.load(os.path.join(hands_folder, f)) for f in os.listdir(hands_folder) if f.endswith(".npy")]), cutoff_frequency, fs_original, fs_new)

# Fit PCA models using data from lips_folder and hands_folder
lips_pca, lips_scaler = fit_pca(lips_data_for_pca, scaler_check=True)
hands_pca, hands_scaler = fit_pca(hands_data_for_pca, scaler_check=True)

# Process each file in syn_folder, transform using PCA, and save results
process_and_transform(syn_folder, lips_pca, hands_pca, lips_scaler, hands_scaler, output_folder, cutoff_frequency, fs_original, fs_new)
