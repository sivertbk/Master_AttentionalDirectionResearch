import mne
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.config import DATASETS, EEG_SETTINGS, PLOTS_PATH, set_plot_style
from utils.file_io import load_raw_data
from utils.helpers import iterate_dataset_items
from utils.preprocessing_tools import fix_bad_channels, prepare_raw_data

# Set up seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
set_plot_style()


def load_and_prepare_data(dataset_config, subject, session, task, bad_channel, adjacent_channels, scale=1e6):
    """
    Load and prepare EEG data for interpolation analysis.
    
    Parameters:
    -----------
    dataset_config : dict
        Dataset configuration
    subject : int
        Subject ID
    session : int
        Session number
    task : str
        Task name 
    bad_channel : str
        Name of the bad channel to interpolate
    adjacent_channels : list
        List of adjacent channel names
    scale : float
        Scaling factor for microvolts conversion
        
    Returns:
    --------
    dict : Dictionary containing all prepared data
    """
    # Prepare the data
    raw = load_raw_data(dataset_config, subject, session=session, task=task, verbose=False)
    raw = prepare_raw_data(raw, dataset_config, EEG_SETTINGS)
    raw.filter(l_freq=1.0, h_freq=None)  # Apply a low-pass filter for better visualization

    t, x_pre = raw.times, raw.get_data(picks=bad_channel) * scale

    # Find the quartiles of the data for y-axis limits
    q1 = np.percentile(x_pre, 25)
    q3 = np.percentile(x_pre, 75)
    y_min = q1 - 1.5 * (q3 - q1)
    y_max = q3 + 1.5 * (q3 - q1)

    # Get adjacent channels data before interpolation
    x_adj1_pre = raw.get_data(picks=adjacent_channels[0]) * scale
    x_adj2_pre = raw.get_data(picks=adjacent_channels[1]) * scale

    raw_post = raw.copy()

    # Fix bad channel
    fix_bad_channels(raw_post, dataset_config, subject, session, task)

    x_post = raw_post.get_data(picks=bad_channel) * scale
    x_adj1_post = raw_post.get_data(picks=adjacent_channels[0]) * scale
    x_adj2_post = raw_post.get_data(picks=adjacent_channels[1]) * scale

    return {
        'raw_pre': raw.copy(),
        'raw_post': raw_post.copy(),
        'bad_channel': bad_channel,
        'times': t,
        'x_pre': x_pre,
        'x_post': x_post,
        'x_adj1_pre': x_adj1_pre,
        'x_adj1_post': x_adj1_post,
        'x_adj2_pre': x_adj2_pre,
        'x_adj2_post': x_adj2_post,
        'y_limits': (y_min, y_max),
        'sampling_freq': raw.info['sfreq']
    }


def extract_time_window(data_dict, start_time=600, stop_time=601):
    """
    Extract a specific time window from the data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the data from load_and_prepare_data
    start_time : float
        Start time in seconds
    stop_time : float
        Stop time in seconds
        
    Returns:
    --------
    dict : Dictionary with windowed data
    """
    fs = data_dict['sampling_freq']
    start_sample = int(start_time * fs)
    stop_sample = int(stop_time * fs)
    
    t = data_dict['times'][start_sample:stop_sample]
    
    return {
        'times': t,
        'x_pre': data_dict['x_pre'][:, start_sample:stop_sample],
        'x_post': data_dict['x_post'][:, start_sample:stop_sample],
        'x_adj1_pre': data_dict['x_adj1_pre'][:, start_sample:stop_sample],
        'x_adj1_post': data_dict['x_adj1_post'][:, start_sample:stop_sample],
        'x_adj2_pre': data_dict['x_adj2_pre'][:, start_sample:stop_sample],
        'x_adj2_post': data_dict['x_adj2_post'][:, start_sample:stop_sample],
        'y_limits': data_dict['y_limits'],
        'fs': fs
    }


def create_interpolation_plot(windowed_data, dataset_config, subject, session, bad_channel, adjacent_channels):
    """
    Create the interpolation analysis plot.
    
    Parameters:
    -----------
    windowed_data : dict
        Dictionary containing windowed data
    dataset_config : dict
        Dataset configuration
    subject : int
        Subject ID
    session : int
        Session number
    bad_channel : str
        Name of the bad channel
    adjacent_channels : list
        List of adjacent channel names
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    t = windowed_data['times']
    x_pre = windowed_data['x_pre']
    x_post = windowed_data['x_post']
    x_adj1_post = windowed_data['x_adj1_post']
    x_adj2_post = windowed_data['x_adj2_post']
    y_min, y_max = windowed_data['y_limits']

    # Create 3 subplots with seaborn styling
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Set overall figure title
    fig.suptitle(f'EEG Channel Interpolation Analysis\n'
                 f'Dataset: {dataset_config.name} | Subject: {subject} | Session: {session}\n',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot first adjacent channel (top)
    axes[0].plot(t, x_adj1_post.T, color='k', linewidth=2, alpha=0.8, label=f'{adjacent_channels[0]}')
    axes[0].set_title(f'Adjacent Channel: {adjacent_channels[0]}', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
    axes[0].set_ylim(y_min, y_max)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=11, frameon=True)
    
    # Plot bad channel (middle) - with both pre and post interpolation
    axes[1].plot(t, x_pre.T, color="#C44747", linewidth=1.5, alpha=0.7, label='Raw (Before Interpolation)', linestyle='--')
    axes[1].plot(t, x_post.T, color="#3EAE85", linewidth=2, alpha=0.9, label='Interpolated (After)')
    axes[1].set_title(f'Bad Channel: {bad_channel} (Before vs After Interpolation)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
    axes[1].set_ylim(y_min, y_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=11, frameon=True)
    
    # Plot second adjacent channel (bottom)
    axes[2].plot(t, x_adj2_post.T, color='k', linewidth=2, alpha=0.8, label=f'{adjacent_channels[1]}')
    axes[2].set_title(f'Adjacent Channel: {adjacent_channels[1]}', fontsize=14, fontweight='bold', pad=15)
    axes[2].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
    axes[2].set_ylim(y_min, y_max)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=11, frameon=True)
    
    # Adjust layout and styling
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for the main title
    
    # Add a subtle background color to the figure
    fig.patch.set_facecolor('#FAFAFA')
    
    return fig

def create_psd_plot(raw_pre, raw_post, dataset_config, subject, session, bad_channel, fmin=1, fmax=40):
    """
    Create the Power Spectral Density (PSD) plot comparing before and after interpolation.
    
    Parameters:
    -----------
    raw_pre : mne.io.Raw
        The raw EEG data before interpolation
    raw_post : mne.io.Raw
        The raw EEG data after interpolation
    dataset_config : dict
        Dataset configuration
    subject : int
        Subject ID
    session : int
        Session number
    bad_channel : str
        Name of the bad channel
    fmin : float
        Minimum frequency for PSD computation
    fmax : float
        Maximum frequency for PSD computation

    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    # Compute PSD for both pre and post interpolation
    psd_pre = raw_pre.compute_psd(fmin=fmin, fmax=fmax, picks=bad_channel)
    psd_post = raw_post.compute_psd(fmin=fmin, fmax=fmax, picks=bad_channel)
    
    freqs = psd_pre.freqs
    
    # Get PSD data and convert to dB
    psd_pre_data = psd_pre.get_data().squeeze()  # Remove singleton dimensions
    psd_post_data = psd_post.get_data().squeeze()
    
    psd_pre_db = 10 * np.log10(psd_pre_data)
    psd_post_db = 10 * np.log10(psd_post_data)
    
    # Determine good ylim based on both signals
    psd_all = np.concatenate([psd_pre_db, psd_post_db])
    lower = np.floor(psd_all.min() - 2)
    upper = np.ceil(psd_all.max() + 2)
    
    # Create the plot with seaborn styling
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot pre-interpolation first (should have more power/artifacts)
    ax.plot(freqs, psd_pre_db, color="#C44747", alpha=0.8, linewidth=1.5, 
            label='Before Interpolation', linestyle='-')
    
    # Plot post-interpolation second (should be cleaner)
    ax.plot(freqs, psd_post_db, color="#000000", alpha=0.9, linewidth=2, 
            label='After Interpolation')
    
    # Fill between the curves to show the difference
    ax.fill_between(freqs, psd_pre_db, psd_post_db, color='gray', alpha=0.25, 
                    label='Difference')
    
    # Styling and labels
    ax.set_ylim([lower, upper])
    ax.set_title(f'Power Spectral Density Comparison - Channel {bad_channel}\n'
                 f'Dataset: {dataset_config.name} | Subject: {subject} | Session: {session}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Spectral Density (dB/Hz)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Add frequency band annotations
    ax.axvspan(1, 4, alpha=0.1, color='purple', label='Delta (1-4 Hz)')
    ax.axvspan(4, 8, alpha=0.1, color='blue', label='Theta (4-8 Hz)')
    ax.axvspan(8, 12, alpha=0.1, color='green', label='Alpha (8-12 Hz)')
    ax.axvspan(12, 30, alpha=0.1, color='orange', label='Beta (12-30 Hz)')
    ax.axvspan(30, fmax, alpha=0.1, color='red', label='Gamma (30- Hz)')

    # Add a text box with statistics
    power_reduction = np.mean(psd_pre_db - psd_post_db)
    delta_power_reduction = np.mean(psd_pre_db[(freqs >= 1) & (freqs < 4)] - psd_post_db[(freqs >= 1) & (freqs < 4)])
    theta_power_reduction = np.mean(psd_pre_db[(freqs >= 4) & (freqs < 8)] - psd_post_db[(freqs >= 4) & (freqs < 8)])
    alpha_power_reduction = np.mean(psd_pre_db[(freqs >= 8) & (freqs < 12)] - psd_post_db[(freqs >= 8) & (freqs < 12)])
    beta_power_reduction = np.mean(psd_pre_db[(freqs >= 12) & (freqs < 30)] - psd_post_db[(freqs >= 12) & (freqs < 30)])
    gamma_power_reduction = np.mean(psd_pre_db[(freqs >= 30) & (freqs <= fmax)] - psd_post_db[(freqs >= 30) & (freqs <= fmax)])
    textstr = f'Avg Power Reduction: {power_reduction:.2f} dB\n' \
              f'Delta: {delta_power_reduction:.2f} dB\n' \
              f'Theta: {theta_power_reduction:.2f} dB\n' \
              f'Alpha: {alpha_power_reduction:.2f} dB\n' \
              f'Beta: {beta_power_reduction:.2f} dB\n' \
              f'Gamma: {gamma_power_reduction:.2f} dB'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Add a subtle background color to the figure
    fig.patch.set_facecolor('#FAFAFA')
    
    return fig

def save_plot(fig, dataset_config, subject, session, title=None):
    """
    Save the plot to the specified path in both SVG and PNG formats.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    dataset_config : dict
        Dataset configuration
    subject : int
        Subject ID
    session : int
        Session number
    title : str, optional
        Title of the plot (used for naming the file)
    """
    title = title or "channel_interpolation"
    base_path = os.path.join(PLOTS_PATH, 'thesis_figures', 'results', 'interpolation')
    filename = f'{title}_{dataset_config.f_name}_{subject}_session{session}'
    os.makedirs(base_path, exist_ok=True)
    svg_path = os.path.join(base_path, f'{filename}.svg')
    png_path = os.path.join(base_path, f'{filename}.png')
    print(f"Saving plot to {svg_path} and {png_path}")
    fig.savefig(svg_path, format='svg', dpi=400)
    fig.savefig(png_path, format='png', dpi=400)
    plt.close(fig)



if __name__ == "__main__":
    # Configuration parameters
    dataset = 'braboszcz2017'
    subject = '070'
    session = None
    task = 'think2'
    bad_channel = 'PO4'
    adjacent_channels = ['PO8', 'O2']
    scale = 1e6  # Scale factor for EEG data in microvolts
    
    # Time window for plotting
    start_time = 500  # Start at 600 seconds
    stop_time = 501   # Stop at 601 seconds

    # Get dataset configuration
    dataset_config = DATASETS.copy()[dataset]

    # Load and prepare data
    print("Loading and preparing data...")
    data = load_and_prepare_data(dataset_config, subject, session, task, bad_channel, adjacent_channels, scale)
    
    # Extract time window
    print(f"Extracting time window: {start_time}-{stop_time} seconds")
    windowed_data = extract_time_window(data, start_time, stop_time)
    
    # Create the plot
    print("Creating interpolation plot...")
    fig = create_interpolation_plot(windowed_data, dataset_config, subject, session, bad_channel, adjacent_channels)
    
    
    # Save the plot
    save_plot(fig, dataset_config, subject, session)

    # Make PSD plot
    print("Creating PSD plot...")
    fig_psd = create_psd_plot(data['raw_pre'], data['raw_post'], dataset_config, subject, session, bad_channel)
    save_plot(fig_psd, dataset_config, subject, session, title='psd')

    print("Analysis complete!")
