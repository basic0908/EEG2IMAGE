import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from tqdm import tqdm


def plot_eeg(X, instance=0):
    """
    Plot the first instance of EEG data with two channels, flexible for any time length.

    Parameters:
    X (numpy.ndarray): EEG data of shape (n_instances, n_timepoints, 2).

    Returns:
    None: Displays a plot of the first instance with time points on the x-axis.
    """
    if X.ndim != 3 or X.shape[2] != 2:
        print("Input shape must be (n_instances, n_timepoints, 2).")
        return
    
    first_instance = X[0]  # Shape (n_timepoints, 2)
    time_axis = np.arange(first_instance.shape[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, first_instance[:, 0], label='Channel 1', marker='o')
    plt.plot(time_axis, first_instance[:, 1], label='Channel 2', marker='o')
    
    plt.xlabel('Timepoints')
    plt.ylabel('Amplitude')
    plt.title('EEG Data Visualization - First Instance')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



def plot_eeg_1280(X, sample_index=0, channels=(0, 1), lim=600):
    """
    Plot EEG data along the 1280 timepoints for two selected channels of a specific sample.
    
    Parameters:
    X (numpy.ndarray): EEG data of shape (230, 14, 1280).
    sample_index (int): Index of the sample to s. Default is 0.
    channels (tuple): Tuple of two channel indices to plot. Default is (0, 1).

    Returns:
    None: Displays a plot for the two selected channels with x-axis limited to lim timepoints.
    """
    n_samples, n_channels, n_timepoints = X.shape
    if sample_index >= n_samples:
        print("Sample index out of range.")
        return
    if any(ch >= n_channels for ch in channels):
        print("Channel index out of range.")
        return
    
    time_axis = np.arange(n_timepoints)
    
    plt.figure(figsize=(12, 6))
    for channel in channels:
        plt.plot(time_axis[:lim], X[sample_index, channel, :lim], label=f'Channel {channel + 1}')
    
    plt.xlabel(f'Timepoints (0-{lim})')
    plt.ylabel('Amplitude')
    plt.title(f'EEG Data Visualization - Sample {sample_index}, Channels {channels[0] + 1} & {channels[1] + 1}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def downsample(X, original_frequency=600, target_frequency=128):
    """
    Downsample EEG data to a target frequency using resampling, with tqdm for progress tracking.

    Parameters:
    X (numpy.ndarray): EEG data of shape (n_instances, n_channels, n_datapoints).
    original_frequency (int): Original sampling frequency (e.g., 600 Hz).
    target_frequency (int): Target sampling frequency (e.g., 128 Hz).

    Returns:
    numpy.ndarray: Downsampled EEG data with adjusted datapoints.
    """
    n_instances, n_channels, n_datapoints = X.shape
    # Calculate the target number of datapoints
    target_datapoints = int(n_datapoints * target_frequency / original_frequency)
    
    # Initialize an array for the downsampled data
    downsampled_data = np.zeros((n_instances, n_channels, target_datapoints))
    
    # Loop through each instance and channel to perform resampling
    for i in tqdm(range(n_instances), desc="Downsampling EEG Data"):
        for channel in range(n_channels):
            downsampled_data[i, channel, :] = resample(X[i, channel, :], target_datapoints)
    
    return downsampled_data

import numpy as np
import matplotlib.pyplot as plt

def plot_combined_correlations(X_VIE, Y_VIE, X_EMOTIV, Y_EMOTIV, vie_labels=['Left', 'Right'], emotiv_labels=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']):
    """
    Calculate and plot combined correlations between channels in VIE and EMOTIV datasets.

    Parameters:
        X_VIE (numpy.ndarray): EEG data from VIE with shape (instances, channels, data points).
        Y_VIE (numpy.ndarray): Labels for VIE data with shape (instances,).
        X_EMOTIV (numpy.ndarray): EEG data from EMOTIV with shape (instances, channels, data points).
        Y_EMOTIV (numpy.ndarray): Labels for EMOTIV data with shape (instances,).
        vie_labels (list): Labels for VIE channels.
        emotiv_labels (list): Labels for EMOTIV channels.

    Returns:
        None
    """
    n_vie_channels = X_VIE.shape[1]
    n_emotiv_channels = X_EMOTIV.shape[1]
    classes = np.unique(Y_VIE)  # Assuming both datasets share the same class labels

    # Initialize a matrix to store combined correlations
    combined_correlations = np.zeros((n_vie_channels, n_emotiv_channels))

    for cls in classes:
        # Filter data for the current class
        X_VIE_cls = X_VIE[Y_VIE == cls]
        X_EMOTIV_cls = X_EMOTIV[Y_EMOTIV == cls]

        # Align the number of instances
        min_instances = min(X_VIE_cls.shape[0], X_EMOTIV_cls.shape[0])
        X_VIE_cls = X_VIE_cls[:min_instances]
        X_EMOTIV_cls = X_EMOTIV_cls[:min_instances]

        # Calculate correlations for the current class
        for vie_channel in range(n_vie_channels):
            for emotiv_channel in range(n_emotiv_channels):
                vie_data = X_VIE_cls[:, vie_channel, :].reshape(-1)
                emotiv_data = X_EMOTIV_cls[:, emotiv_channel, :].reshape(-1)
                correlation = np.corrcoef(vie_data, emotiv_data)[0, 1]
                combined_correlations[vie_channel, emotiv_channel] += correlation

    # Take the average of the correlations across classes
    combined_correlations /= len(classes)

    # Plot the combined correlations as a bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(n_emotiv_channels)

    for vie_channel in range(n_vie_channels):
        ax.bar(index + vie_channel * bar_width, combined_correlations[vie_channel], bar_width, label=vie_labels[vie_channel])

    ax.set_xlabel('EMOTIV Channels')
    ax.set_ylabel('Average Correlation Coefficient')
    ax.set_title('Average Correlations Between VIE and EMOTIV Channels')
    ax.set_xticks(index + (bar_width * (n_vie_channels - 1) / 2))
    ax.set_xticklabels(emotiv_labels, rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()