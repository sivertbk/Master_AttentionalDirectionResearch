import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from utils.config import DATASETS, PLOTS_PATH
from eeg_analyzer.dataset import Dataset
from eeg_analyzer.metrics import Metrics

dataset_name = "braboszcz2017"
subject_id = "076"
session_id = 1

save_dir = os.path.join(PLOTS_PATH, dataset_name, "density_plot")
os.makedirs(save_dir, exist_ok=True)

dataset_config = DATASETS[dataset_name]
dataset = Dataset(dataset_config)

dataset.load_subjects()

subject = dataset.get_subject(subject_id)
print(subject)

rec = subject.get_recording(session_id=session_id)
print(rec)

med_psd = rec.get_psd("med2", "OT")
med_freqs = rec.get_freqs("med2", "OT")
print(f"med2 shape: {med_psd.shape}")
think_psd = rec.get_psd("think2", "MW")
think_freqs = rec.get_freqs("think2", "MW")
print(f"think2 shape: {think_psd.shape}")

med_alpha = Metrics.alpha_power(med_psd, med_freqs)
think_alpha = Metrics.alpha_power(think_psd, think_freqs)
print(f"med2 alpha power: {med_alpha.shape}")
print(f"think2 alpha power: {think_alpha.shape}")

# Convert values to dB


# I have shape (epochs, channels) for both med_alpha and think_alpha
# I want to plot the distribution of alpha power for each channel

channel_names = rec.get_channel_names()

# Prepare data for Seaborn
data_med = []
for i, ch_name in enumerate(channel_names):
    for epoch_power in med_alpha[:, i]:
        data_med.append({'alpha_power': epoch_power, 'condition': 'med2_OT', 'channel': ch_name})
df_med = pd.DataFrame(data_med)

data_think = []
for i, ch_name in enumerate(channel_names):
    for epoch_power in think_alpha[:, i]:
        data_think.append({'alpha_power': epoch_power, 'condition': 'think2_MW', 'channel': ch_name})
df_think = pd.DataFrame(data_think)

df_combined = pd.concat([df_med, df_think], ignore_index=True)

# Plotting
for channel_name in channel_names:
    plt.figure(figsize=(10, 6))
    
    channel_data = df_combined[df_combined['channel'] == channel_name]
    
    sns.histplot(
        data=channel_data[channel_data['condition'] == 'med2_OT'],
        x='alpha_power',
        kde=True,
        label='med2_OT',
        stat='density',
        alpha=0.6,
        color='blue'
    )
    
    sns.histplot(
        data=channel_data[channel_data['condition'] == 'think2_MW'],
        x='alpha_power',
        kde=True,
        label='think2_MW',
        stat='density',
        alpha=0.6,
        color='red'
    )
    
    plt.title(f'Alpha Power Distribution for Channel {channel_name} (Subject {subject.id})')
    plt.xlabel('Alpha Power (µV²/Hz)')
    plt.ylabel('Density')
    plt.legend(title='Condition')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_filename = os.path.join(save_dir, f"s{subject.id}_channel_{channel_name}_alpha_dist.png")
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")
    plt.close()

print(f"All plots saved to {save_dir}")