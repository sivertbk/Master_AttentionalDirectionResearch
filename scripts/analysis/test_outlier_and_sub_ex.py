"""
Comprehensive Testing Script for Outlier Detection and Subject Exclusion
========================================================================

This script tests the functionality of the outlier detection and subject exclusion 
mechanisms implemented in the Recording class. It creates visualizations showing:

1. Data distributions before and after log transformation
2. Data distributions before and after outlier detection
3. Summary statistics and quality control metrics
4. Comparison of different outlier detection parameters

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import pandas as pd
from typing import Tuple, Dict, Any

from eeg_analyzer.subject import Subject
from utils.config import DATASETS, PLOTS_PATH, OUTLIER_DETECTION, QUALITY_CONTROL, set_plot_style

# Configure plotting style
set_plot_style()
sns.set_palette("husl")

def create_output_directory(base_path: str, subdir: str) -> str:
    """Create output directory for plots."""
    output_path = os.path.join(base_path, subdir)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def plot_distribution_comparison(data_before: np.ndarray, data_after: np.ndarray, 
                               title: str, xlabel: str, save_path: str = None) -> None:
    """Plot distribution comparison before and after transformation/filtering."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Before transformation - histogram
    axes[0, 0].hist(data_before.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Before - Histogram')
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # After transformation - histogram
    axes[0, 1].hist(data_after.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('After - Histogram')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Before transformation - Q-Q plot
    stats.probplot(data_before.flatten(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Before - Q-Q Plot (Normal)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # After transformation - Q-Q plot
    stats.probplot(data_after.flatten(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('After - Q-Q Plot (Normal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.show()

def plot_outlier_detection_summary(outlier_summary: Dict, save_path: str = None) -> None:
    """Plot summary of outlier detection results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Outlier Detection Summary', fontsize=14, fontweight='bold')
    
    # Outlier percentages by condition
    if 'outlier_percentages' in outlier_summary:
        conditions = list(outlier_summary['outlier_percentages'].keys())
        percentages = list(outlier_summary['outlier_percentages'].values())
        
        axes[0, 0].bar(range(len(conditions)), percentages, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Outlier Percentage by Condition')
        axes[0, 0].set_xlabel('Condition')
        axes[0, 0].set_ylabel('Outlier Percentage (%)')
        axes[0, 0].set_xticks(range(len(conditions)))
        axes[0, 0].set_xticklabels([f"{c[0]}-{c[1]}" for c in conditions], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    
    # Overall statistics
    stats_text = f"""Total Original Epochs: {outlier_summary.get('total_epochs_original', 'N/A')}
Total Epochs Removed: {outlier_summary.get('total_epochs_removed', 'N/A')}
Overall Outlier %: {outlier_summary.get('overall_outlier_percentage', 'N/A'):.2f}%
Conditions Processed: {outlier_summary.get('conditions_processed', 'N/A')}"""
    
    axes[0, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title('Summary Statistics')
    axes[0, 1].axis('off')
    
    # Distribution of outlier percentages
    if 'outlier_percentages' in outlier_summary:
        axes[1, 0].hist(percentages, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Distribution of Outlier Percentages')
        axes[1, 0].set_xlabel('Outlier Percentage (%)')
        axes[1, 0].set_ylabel('Number of Conditions')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Configuration used
    config_text = f"""IQR Multiplier: {OUTLIER_DETECTION['IQR_MULTIPLIER']}
Skewness Threshold: {OUTLIER_DETECTION['SKEWNESS_THRESHOLD']}
Min Epochs for Filtering: {OUTLIER_DETECTION['MIN_EPOCHS_FOR_FILTERING']}
Suspicious Skewness: {OUTLIER_DETECTION['SUSPICIOUS_SKEWNESS_THRESHOLD']}
Suspicious Kurtosis: {OUTLIER_DETECTION['SUSPICIOUS_KURTOSIS_THRESHOLD']}"""
    
    axes[1, 1].text(0.1, 0.5, config_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Configuration Parameters')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.show()

def plot_quality_control_summary(recording, save_path: str = None) -> None:
    """Plot quality control metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quality Control Summary', fontsize=14, fontweight='bold')
    
    # Get quality control summary
    qc_summary = recording.get_quality_control_summary()
    
    # MW/OT epoch ratio
    mw_ot_ratio = recording.get_mw_ot_epoch_ratio()
    axes[0, 0].bar(['MW/OT Ratio'], [mw_ot_ratio], color='green' if mw_ot_ratio != np.inf else 'red')
    axes[0, 0].set_title('MW/OT Epoch Ratio')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    if mw_ot_ratio != np.inf:
        axes[0, 0].axhline(y=QUALITY_CONTROL['STATE_RATIO_THRESHOLD'], color='red', 
                          linestyle='--', label=f"Threshold: {QUALITY_CONTROL['STATE_RATIO_THRESHOLD']}")
        axes[0, 0].legend()
    
    # Epoch counts by condition
    epoch_counts = recording.get_num_epochs()
    conditions = list(epoch_counts.keys())
    counts = list(epoch_counts.values())
    
    axes[0, 1].bar(range(len(conditions)), counts, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Epoch Counts by Condition')
    axes[0, 1].set_xlabel('Condition')
    axes[0, 1].set_ylabel('Number of Epochs')
    axes[0, 1].set_xticks(range(len(conditions)))
    axes[0, 1].set_xticklabels([f"{c[0]}-{c[1]}" for c in conditions], rotation=45)
    axes[0, 1].axhline(y=QUALITY_CONTROL['MIN_EPOCHS_PER_STATE'], color='red', 
                      linestyle='--', label=f"Min Required: {QUALITY_CONTROL['MIN_EPOCHS_PER_STATE']}")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quality control status text
    qc_text = f"""State Imbalance: {qc_summary.get('state_imbalance', 'N/A')}
Minimum Epochs Check: {qc_summary.get('minimum_epochs', 'N/A')}
Should Exclude: {qc_summary.get('should_exclude', 'N/A')}
MW/OT Ratio: {mw_ot_ratio:.3f}
Threshold: {QUALITY_CONTROL['STATE_RATIO_THRESHOLD']:.3f}"""
    
    axes[1, 0].text(0.1, 0.5, qc_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Quality Control Status')
    axes[1, 0].axis('off')
    
    # Configuration parameters
    config_text = f"""State Ratio Threshold: {QUALITY_CONTROL['STATE_RATIO_THRESHOLD']}
Min Epochs per State: {QUALITY_CONTROL['MIN_EPOCHS_PER_STATE']}
MW/OT States: {QUALITY_CONTROL['MW_OT_STATES']}
Recording Exclude Flag: {recording.exclude}"""
    
    axes[1, 1].text(0.1, 0.5, config_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Configuration & Status')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.show()

def test_outlier_detection_and_quality_control():
    """Main testing function."""
    
    print("="*80)
    print("TESTING OUTLIER DETECTION AND SUBJECT EXCLUSION FUNCTIONALITY")
    print("="*80)
    
    # Setup
    dataset_f_name = 'jin2019'
    dataset_config = DATASETS[dataset_f_name]
    output_path = create_output_directory(PLOTS_PATH, f"{dataset_config.f_name}_outlier_testing")
    
    print(f"\nDataset: {dataset_config.name}")
    print(f"Output directory: {output_path}")
    
    # Load subject and recording
    print(f"\nLoading subject data...")
    subject = Subject(dataset_config, "2")
    subject.load_data()
    
    print(f"Loaded data for subject {subject.get_id()} from dataset {subject.get_dataset().name}")
    print(f"Subject has {subject.get_total_epochs()} epochs.")
    
    # Get recording and calculate band power
    recording = subject.get_recording(session_id=1)
    recording.calculate_band_power((8, 12))  # Alpha band
    
    print(f"Recording has {len(recording.list_conditions())} conditions")
    print(f"Available conditions: {recording.list_conditions()}")
    
    # 1. Test data distributions before and after log transformation
    print(f"\n1. Testing data distributions and log transformation...")
    
    # Get some sample data (alpha band power)
    if recording.list_conditions():
        task, state = recording.list_conditions()[0]
        original_data = recording.get_band_power(task, state)
        log_transformed_data = np.log10(original_data + 1e-10)  # Add small constant to avoid log(0)
        
        plot_distribution_comparison(
            original_data, log_transformed_data,
            f'Data Distribution: {task}-{state} (Alpha Band Power)',
            'Power (μV²/Hz)',
            os.path.join(output_path, f'distribution_comparison_{task}_{state}.png')
        )
        
        print(f"Original data - Mean: {np.mean(original_data):.4f}, Std: {np.std(original_data):.4f}")
        print(f"Log-transformed - Mean: {np.mean(log_transformed_data):.4f}, Std: {np.std(log_transformed_data):.4f}")
    
    # 2. Test outlier detection
    print(f"\n2. Testing outlier detection...")
    
    # Store original data
    original_band_power_map = {}
    for task, states in recording.band_power_map.items():
        original_band_power_map[task] = {}
        for state, data in states.items():
            original_band_power_map[task][state] = data.copy()
    
    # Apply outlier filtering
    print("Applying outlier filtering...")
    recording.apply_outlier_filtering(data_type='band_power')
    
    # Get outlier summary
    outlier_summary = recording.get_outlier_summary()
    print(f"Outlier detection summary:")
    for key, value in outlier_summary.items():
        print(f"  {key}: {value}")
    
    # Plot outlier detection results
    plot_outlier_detection_summary(
        outlier_summary,
        os.path.join(output_path, 'outlier_detection_summary.png')
    )
    
    # 3. Compare distributions before and after outlier removal
    print(f"\n3. Comparing distributions before and after outlier removal...")
    
    if recording.list_conditions():
        task, state = recording.list_conditions()[0]
        if task in original_band_power_map and state in original_band_power_map[task]:
            original_data = original_band_power_map[task][state]
            filtered_data = recording.get_band_power(task, state)
            
            plot_distribution_comparison(
                original_data, filtered_data,
                f'Outlier Detection: {task}-{state} (Alpha Band Power)',
                'Power (μV²/Hz)',
                os.path.join(output_path, f'outlier_detection_{task}_{state}.png')
            )
            
            print(f"Before outlier removal - Shape: {original_data.shape}, Mean: {np.mean(original_data):.4f}")
            print(f"After outlier removal - Shape: {filtered_data.shape}, Mean: {np.mean(filtered_data):.4f}")
            print(f"Epochs removed: {original_data.shape[0] - filtered_data.shape[0]}")
    
    # 4. Test suspicious distribution detection
    print(f"\n4. Testing suspicious distribution detection...")
    
    suspicious_flags = recording.detect_suspicious_distributions()
    print("Suspicious distribution flags:")
    for key, flags in suspicious_flags.items():
        if any(flags.values()):
            print(f"  {key}: {flags}")
    
    # 5. Test quality control
    print(f"\n5. Testing quality control functionality...")
    
    # Get quality control summary
    qc_summary = recording.get_quality_control_summary()
    print("Quality control summary:")
    for key, value in qc_summary.items():
        print(f"  {key}: {value}")
    
    # Plot quality control summary
    plot_quality_control_summary(
        recording,
        os.path.join(output_path, 'quality_control_summary.png')
    )
    
    # 6. Test different parameter configurations
    print(f"\n6. Testing different outlier detection parameters...")
    
    # Test with more stringent parameters
    print("Testing with more stringent parameters...")
    recording_strict = subject.get_recording(session_id=1)
    recording_strict.calculate_band_power((8, 12))
    
    # Apply more stringent filtering
    recording_strict.apply_outlier_filtering(
        data_type='band_power',
        k=1.5,  # More stringent IQR multiplier
        s=1.5   # More stringent skewness threshold
    )
    
    strict_summary = recording_strict.get_outlier_summary()
    print("Strict parameters outlier summary:")
    for key, value in strict_summary.items():
        print(f"  {key}: {value}")
    
    # Compare results
    print(f"\nComparison of outlier detection:")
    print(f"Default parameters - Total removed: {outlier_summary.get('total_epochs_removed', 0)}")
    print(f"Strict parameters - Total removed: {strict_summary.get('total_epochs_removed', 0)}")
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETED SUCCESSFULLY!")
    print(f"All plots saved to: {output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_outlier_detection_and_quality_control()

