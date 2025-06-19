import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import pandas as pd
from typing import Tuple, Dict, Any
from collections import defaultdict, Counter

from eeg_analyzer.dataset import Dataset
from utils.config import DATASETS, PLOTS_PATH, OUTLIER_DETECTION, QUALITY_CONTROL, set_plot_style

# Configure plotting style
set_plot_style()

def analyze_suspicious_distributions(dataset_name: str, dataset: Dataset) -> Dict[str, Any]:
    """
    Analyze suspicious epoch,channel pairs for a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object with loaded subjects
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"\nAnalyzing suspicious distributions for {dataset_name}...")
    
    # Collect all suspicious flags from all subjects and recordings
    all_suspicious_flags = defaultdict(lambda: defaultdict(list))
    subject_recording_count = 0
    
    # Statistics containers
    flag_counts = Counter()
    channel_flag_counts = defaultdict(Counter)
    condition_flag_counts = defaultdict(Counter)
    
    for subject_id, subject in dataset.subjects.items():
        for session_id, recording in subject.recordings.items():
            subject_recording_count += 1
            
            # Get suspicious distributions for this recording
            suspicious_flags = recording.detect_suspicious_distributions()
            
            # Process each data type (band_power, log_band_power)
            for data_type, type_flags in suspicious_flags.items():
                # Process all_data level
                for channel, flags in type_flags['all_data'].items():
                    if flags:  # If there are any flags
                        for flag in flags:
                            flag_key = f"{data_type}_all_{flag}"
                            flag_counts[flag_key] += 1
                            channel_flag_counts[channel][flag_key] += 1
                            all_suspicious_flags[data_type]['all_data'].append({
                                'subject_id': subject_id,
                                'session_id': session_id,
                                'channel': channel,
                                'flags': flags
                            })
                
                # Process by_state level
                for state, state_channels in type_flags['by_state'].items():
                    for channel, flags in state_channels.items():
                        if flags:
                            for flag in flags:
                                flag_key = f"{data_type}_{state}_{flag}"
                                flag_counts[flag_key] += 1
                                channel_flag_counts[channel][flag_key] += 1
                                all_suspicious_flags[data_type]['by_state'].append({
                                    'subject_id': subject_id,
                                    'session_id': session_id,
                                    'channel': channel,
                                    'state': state,
                                    'flags': flags
                                })
                
                # Process by_condition level
                for condition, condition_channels in type_flags['by_condition'].items():
                    for channel, flags in condition_channels.items():
                        if flags:
                            task, state = condition
                            for flag in flags:
                                flag_key = f"{data_type}_{task}_{state}_{flag}"
                                flag_counts[flag_key] += 1
                                channel_flag_counts[channel][flag_key] += 1
                                condition_flag_counts[condition][flag_key] += 1
                                all_suspicious_flags[data_type]['by_condition'].append({
                                    'subject_id': subject_id,
                                    'session_id': session_id,
                                    'channel': channel,
                                    'task': task,
                                    'state': state,
                                    'flags': flags
                                })
    
    # Calculate summary statistics
    total_possible_pairs = subject_recording_count * len(recording.channels) if subject_recording_count > 0 else 0
    total_flagged_pairs = sum(flag_counts.values())
    
    results = {
        'dataset_name': dataset_name,
        'subject_count': len(dataset.subjects),
        'recording_count': subject_recording_count,
        'total_possible_pairs': total_possible_pairs,
        'total_flagged_pairs': total_flagged_pairs,
        'flagged_percentage': (total_flagged_pairs / max(total_possible_pairs, 1)) * 100,
        'flag_counts': dict(flag_counts),
        'channel_flag_counts': dict(channel_flag_counts),
        'condition_flag_counts': dict(condition_flag_counts),
        'all_suspicious_flags': dict(all_suspicious_flags),
        'channels': recording.channels if subject_recording_count > 0 else []
    }
    
    return results

def plot_suspicious_distributions(results_dict: Dict[str, Dict[str, Any]]):
    """
    Create comprehensive plots of suspicious distribution analysis.
    
    Args:
        results_dict: Dictionary with dataset names as keys and analysis results as values
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall flag count comparison across datasets
    plt.subplot(3, 3, 1)
    datasets = list(results_dict.keys())
    total_flags = [results_dict[d]['total_flagged_pairs'] for d in datasets]
    flagged_percentages = [results_dict[d]['flagged_percentage'] for d in datasets]
    
    bars = plt.bar(datasets, total_flags, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Total Suspicious Epoch-Channel Pairs by Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Flagged Pairs')
    plt.xticks(rotation=45)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, flagged_percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_flags)*0.01,
                f'{pct:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Flag type distribution
    plt.subplot(3, 3, 2)
    all_flag_types = set()
    for results in results_dict.values():
        all_flag_types.update(results['flag_counts'].keys())
    
    flag_type_counts = defaultdict(list)
    for flag_type in sorted(all_flag_types):
        for dataset in datasets:
            count = results_dict[dataset]['flag_counts'].get(flag_type, 0)
            flag_type_counts[flag_type].append(count)
    
    # Plot only top 10 most common flag types
    sorted_flags = sorted(flag_type_counts.items(), key=lambda x: sum(x[1]), reverse=True)[:10]
    
    x = np.arange(len(datasets))
    width = 0.8 / len(sorted_flags)
    
    for i, (flag_type, counts) in enumerate(sorted_flags):
        offset = (i - len(sorted_flags)/2) * width + width/2
        plt.bar(x + offset, counts, width, label=flag_type.split('_')[-1], alpha=0.8)
    
    plt.title('Flag Type Distribution Across Datasets', fontsize=12, fontweight='bold')
    plt.xlabel('Dataset')
    plt.ylabel('Count')
    plt.xticks(x, datasets, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. Channel-wise suspicious flags (heatmap for each dataset)
    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        plt.subplot(3, 3, 3 + idx)
        
        channels = results['channels']
        if not channels:
            plt.text(0.5, 0.5, f'No data for {dataset_name}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{dataset_name}: Channel Flags', fontsize=10, fontweight='bold')
            continue
            
        # Create matrix of channel vs flag type
        common_flags = ['high_skewness', 'high_kurtosis', 'low_epoch_count', 'non_normal', 'missing_data']
        channel_matrix = np.zeros((len(channels), len(common_flags)))
        
        for ch_idx, channel in enumerate(channels):
            for flag_idx, flag in enumerate(common_flags):
                # Sum across all data types and conditions for this flag type
                total_count = 0
                for flag_key, count in results['channel_flag_counts'][channel].items():
                    if flag in flag_key:
                        total_count += count
                channel_matrix[ch_idx, flag_idx] = total_count
        
        if channel_matrix.max() > 0:
            im = plt.imshow(channel_matrix, cmap='Reds', aspect='auto')
            plt.colorbar(im, shrink=0.6)
        else:
            plt.imshow(np.zeros((len(channels), len(common_flags))), cmap='Reds', aspect='auto')
        
        plt.title(f'{dataset_name}: Channel Flags', fontsize=10, fontweight='bold')
        plt.xlabel('Flag Type')
        plt.ylabel('Channel')
        plt.xticks(range(len(common_flags)), common_flags, rotation=45, fontsize=8)
        plt.yticks(range(len(channels)), channels, fontsize=6)
    
    # 4. Statistics summary table
    plt.subplot(3, 3, 7)
    plt.axis('tight')
    plt.axis('off')
    
    # Create summary table data
    table_data = []
    for dataset_name, results in results_dict.items():
        table_data.append([
            dataset_name,
            f"{results['subject_count']}",
            f"{results['recording_count']}",
            f"{results['total_flagged_pairs']}",
            f"{results['flagged_percentage']:.2f}%"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Dataset', 'Subjects', 'Recordings', 'Flagged Pairs', 'Flagged %'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('Dataset Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # 5. Flag severity distribution
    plt.subplot(3, 3, 8)
    severity_counts = defaultdict(list)
    
    for dataset_name, results in results_dict.items():
        high_severity = 0  # high_skewness, high_kurtosis
        medium_severity = 0  # non_normal, low_epoch_count
        low_severity = 0  # missing_data
        
        for flag_key, count in results['flag_counts'].items():
            if 'high_skewness' in flag_key or 'high_kurtosis' in flag_key:
                high_severity += count
            elif 'non_normal' in flag_key or 'low_epoch_count' in flag_key:
                medium_severity += count
            elif 'missing_data' in flag_key:
                low_severity += count
        
        severity_counts['High'].append(high_severity)
        severity_counts['Medium'].append(medium_severity)
        severity_counts['Low'].append(low_severity)
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    for i, (severity, counts) in enumerate(severity_counts.items()):
        plt.bar(x + i*width, counts, width, label=severity, color=colors[i], alpha=0.8)
    
    plt.title('Flag Severity Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Dataset')
    plt.ylabel('Count')
    plt.xticks(x + width, datasets, rotation=45)
    plt.legend()
    
    # 6. Data quality overview
    plt.subplot(3, 3, 9)
    
    quality_scores = []
    for dataset_name, results in results_dict.items():
        # Calculate a simple quality score (higher is better)
        total_pairs = results['total_possible_pairs']
        flagged_pairs = results['total_flagged_pairs']
        quality_score = max(0, 100 - results['flagged_percentage'])
        quality_scores.append(quality_score)
    
    colors = ['green' if score >= 90 else 'orange' if score >= 70 else 'red' for score in quality_scores]
    bars = plt.bar(datasets, quality_scores, color=colors, alpha=0.7)
    
    plt.title('Data Quality Score', fontsize=12, fontweight='bold')
    plt.ylabel('Quality Score (%)')
    plt.xlabel('Dataset')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # Add score labels on bars
    for bar, score in zip(bars, quality_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOTS_PATH, 'suspicious_distributions_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()

def print_detailed_statistics(results_dict: Dict[str, Dict[str, Any]]):
    """
    Print detailed statistics about suspicious distributions.
    
    Args:
        results_dict: Dictionary with dataset names as keys and analysis results as values
    """
    print("\n" + "="*80)
    print("DETAILED SUSPICIOUS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for dataset_name, results in results_dict.items():
        print(f"\n{dataset_name.upper()} DATASET:")
        print("-" * 40)
        print(f"Subjects: {results['subject_count']}")
        print(f"Recordings: {results['recording_count']}")
        print(f"Total possible epoch-channel pairs: {results['total_possible_pairs']}")
        print(f"Flagged pairs: {results['total_flagged_pairs']}")
        print(f"Percentage flagged: {results['flagged_percentage']:.2f}%")
        
        print(f"\nTop flag types:")
        sorted_flags = sorted(results['flag_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        for flag_type, count in sorted_flags:
            percentage = (count / max(results['total_flagged_pairs'], 1)) * 100
            print(f"  {flag_type}: {count} ({percentage:.1f}% of flagged pairs)")
        
        print(f"\nMost problematic channels:")
        channel_totals = {}
        for channel, flag_counts in results['channel_flag_counts'].items():
            channel_totals[channel] = sum(flag_counts.values())
        
        sorted_channels = sorted(channel_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        for channel, total_flags in sorted_channels:
            if total_flags > 0:
                print(f"  {channel}: {total_flags} flags")
    
    # Cross-dataset comparison
    print(f"\nCROSS-DATASET COMPARISON:")
    print("-" * 40)
    
    all_datasets = list(results_dict.keys())
    flagged_percentages = [results_dict[d]['flagged_percentage'] for d in all_datasets]
    
    best_dataset = all_datasets[np.argmin(flagged_percentages)]
    worst_dataset = all_datasets[np.argmax(flagged_percentages)]
    
    print(f"Best data quality: {best_dataset} ({min(flagged_percentages):.2f}% flagged)")
    print(f"Worst data quality: {worst_dataset} ({max(flagged_percentages):.2f}% flagged)")
    print(f"Average flagged percentage: {np.mean(flagged_percentages):.2f}%")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print("-" * 40)
    
    avg_flagged = np.mean(flagged_percentages)
    if avg_flagged > 20:
        print("⚠️  HIGH: >20% of data flagged as suspicious. Consider:")
        print("   - Reviewing preprocessing pipeline")
        print("   - Adjusting outlier detection thresholds")
        print("   - Investigating data collection procedures")
    elif avg_flagged > 10:
        print("⚠️  MODERATE: 10-20% of data flagged. Consider:")
        print("   - Reviewing most common flag types")
        print("   - Checking specific problematic channels")
    else:
        print("✅ GOOD: <10% of data flagged as suspicious.")
        print("   - Data quality appears acceptable")
        print("   - Continue with standard analysis procedures")

def main():
    """Main function to run the suspicious distribution analysis."""
    print("Starting suspicious distribution analysis for all datasets...")
    print(f"Available datasets: {list(DATASETS.keys())}")
    
    # Dictionary to store results for all datasets
    all_results = {}
    
    # Process each dataset
    for dataset_name, dataset_config in DATASETS.items():
        try:
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name}")
            print(f"{'='*50}")
            
            # Create and load dataset
            dataset = Dataset(dataset_config)
            dataset.load_subjects(variant="mean")
            
            if len(dataset.subjects) == 0:
                print(f"Warning: No subjects loaded for {dataset_name}. Skipping.")
                continue
            
            print(f"Loaded {len(dataset.subjects)} subjects with {sum(len(s.recordings) for s in dataset.subjects.values())} recordings")
            
            # Analyze suspicious distributions
            results = analyze_suspicious_distributions(dataset_name, dataset)
            all_results[dataset_name] = results
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue
    
    if not all_results:
        print("No datasets processed successfully. Exiting.")
        return
    
    # Create plots and print statistics
    print(f"\n{'='*50}")
    print("Creating visualization and summary...")
    print(f"{'='*50}")
    
    plot_suspicious_distributions(all_results)
    print_detailed_statistics(all_results)
    
    print(f"\nAnalysis complete! Results saved to {PLOTS_PATH}")

if __name__ == "__main__":
    main()

