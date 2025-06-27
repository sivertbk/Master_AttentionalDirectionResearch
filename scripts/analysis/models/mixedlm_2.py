import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns  
import mne
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from eeg_analyzer import EEGAnalyzer
from utils.config import EEGANALYZER_SETTINGS

eeganalyzer_kwargs = EEGANALYZER_SETTINGS.copy()
ANALYZER_NAME = EEGANALYZER_SETTINGS["analyzer_name"]
MODEL_NAME = "mixedlm_2"

# Trying to load the EEGAnalyzer
analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
if analyzer is None:
    print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
    analyzer = EEGAnalyzer(**eeganalyzer_kwargs)
    analyzer.save_analyzer()

# Creating a DataFrame with the data
df = analyzer.create_dataframe()

model_path = os.path.join(analyzer.derivatives_path, MODEL_NAME) 
os.makedirs(model_path, exist_ok=True)

# Print out information about the DataFrame
print("DataFrame Information:")
print("="*30)
print(f"Total number of rows: {len(df)}")
print(f"Datasets: {df['dataset'].unique().tolist()}")

for dataset_name in df['dataset'].unique():
    df_dataset = df[df['dataset'] == dataset_name]
    print(f"\n--- Dataset: {dataset_name} ---")
    print(f"  Task orientation: {df_dataset['task_orientation'].iloc[0]}")
    print(f"  Subjects: {df_dataset['subject_id'].nunique()} ({df_dataset['subject_id'].unique().tolist()})")
    print(f"  Sessions: {df_dataset['session_id'].nunique()}")
    print(f"  Channels: {df_dataset['channel'].nunique()}")
    print(f"  Groups: {df_dataset['group'].unique().tolist()}")
    print(f"  States: {df_dataset['state'].unique().tolist()}")
    print(f"  Total data points: {len(df_dataset)}")
    
    print("\n  Data points per channel:")
    print(df_dataset.groupby('channel')['log_band_power'].count())
    
    print("\n  Data points per subject:")
    print(df_dataset.groupby('subject_id')['log_band_power'].count())


# Fitting a linear mixed effects model for each dataset
for dataset_name in df['dataset'].unique():
    
    print(f"\n\nFitting model for dataset: {dataset_name}")
    df_dataset = df[df['dataset'] == dataset_name].copy()

    dataset_model_path = os.path.join(model_path, dataset_name)
    os.makedirs(dataset_model_path, exist_ok=True)

    df_dataset["sub_ch"] = df_dataset["subject_id"].astype(str) + "_" + df_dataset["channel"]     # unique sensor instance

    if dataset_name == "Jin et al. 2019":
        # For this dataset we can have task as variance component
        vc_formula = {
            "subject": "0 + C(subject_id)",  # subject intercepts
            "task": "0 + C(task)"            # task intercepts
        }
    else:
        vc_formula = {
            "subject": "0 + C(subject_id)"   # subject intercepts
        }

    model = smf.mixedlm(
        "log_band_power ~ C(state, Treatment(reference='MW'))",  # MW is the reference state. Result is OT-MW slope
        df_dataset,
        groups=df_dataset["sub_ch"],      # each physical sensor instance gets its own intercept
        re_formula="1 + C(state)",        # random slope for state, *pooled* because channel label is in vc_formula
        vc_formula=vc_formula
    )

    # Save the dataset used here:
    df_dataset.to_csv(os.path.join(dataset_model_path, "dataset_used.csv"), index=False)

    result = model.fit(method='lbfgs')
    print(result.summary())

    random_effects = result.random_effects

    slopes = {}
    for key, effect in random_effects.items():
        slopes[key] = effect["C(state)[T.OT]"]  

    df_slopes = pd.DataFrame({
        'sub_ch': slopes.keys(),
        'slope': slopes.values()
    })
    df_slopes['channel'] = df_slopes['sub_ch'].apply(lambda x: x.split('_')[1]) 

    # Aggregate mean and std
    channel_stats = df_slopes.groupby('channel')['slope'].agg(['mean', 'std', 'count']).reset_index()
    channel_stats['sem'] = channel_stats['std'] / np.sqrt(channel_stats['count'])  # standard error of the mean

    channel_stats['z'] = channel_stats['mean'] / channel_stats['sem']
    channel_stats['p_uncorrected'] = 2 * (1 - norm.cdf(np.abs(channel_stats['z'])))

    reject, p_fdr, _, _ = multipletests(channel_stats['p_uncorrected'], method='fdr_bh')
    channel_stats['p_fdr'] = p_fdr
    channel_stats['significant'] = reject

    # Save random effects
    re_csv_path = os.path.join(dataset_model_path, "random_effects.csv")
    re_data = []
    for key, effect in random_effects.items():
        row = {"sub_ch": key}
        row.update(effect)
        re_data.append(row)
    pd.DataFrame(re_data).to_csv(re_csv_path, index=False)
    print(f"Saved random effects to {re_csv_path}")

    # Save channel_stats
    channel_stats_path = os.path.join(dataset_model_path, "channel_stats.csv")
    channel_stats.to_csv(channel_stats_path, index=False)
    print(f"Saved channel stats to {channel_stats_path}")

    montage = mne.channels.make_standard_montage('biosemi64')  # Use a standard montage for visualization

    # Create info structure
    info = mne.create_info(ch_names=channel_stats['channel'].tolist(), sfreq=256, ch_types='eeg')
    info.set_montage(montage)

    # Values to plot: mean slopes masked by significance
    topo_data = np.array(channel_stats['mean'])
    mask = channel_stats['significant'].values

    fig, ax = plt.subplots()
    im, _ = mne.viz.plot_topomap(topo_data, info, mask=mask, axes=ax, cmap='RdBu_r',
                        contours=0, vlim=(-np.max(np.abs(topo_data)),np.max(np.abs(topo_data))), 
                        mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
                        linewidth=2), show=False)
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Alpha Power [ln(µV²)]', rotation=90, labelpad=15)
    ax.set_title(f"Mixed Effects Model\n{dataset_name} State-Effect Slopes (OT-MW)", fontsize=16)
    plt.savefig(os.path.join(dataset_model_path, "topoplot_slopes.svg"), format='svg', bbox_inches='tight')

    # Creating diagnostic plots
    diag = pd.DataFrame({
        "fitted": result.fittedvalues,             # μ̂
        "resid":  result.resid,                    # raw residuals
    })

    # Standardised / studentised residuals 
    sigma = np.sqrt(result.scale)                 # √(residual variance)
    diag["resid_std"] = diag["resid"] / sigma

    sm.qqplot(diag["resid_std"], line="45", fit=True)
    plt.title("Q–Q Plot of Standardised Residuals")
    plt.savefig(os.path.join(dataset_model_path, "qqplot_residuals.png"), format='png', bbox_inches='tight')


    plt.figure()
    sns.scatterplot(x="fitted", y="resid_std", data=diag, s=8, alpha=.3)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Standardised residuals")
    plt.title("Residuals vs. Fitted Values")
    plt.savefig(os.path.join(dataset_model_path, "residuals_vs_fitted.png"), format='png', bbox_inches='tight')


    sns.histplot(diag["resid_std"], bins=50, kde=True)
    plt.xlabel("Standardised residuals")
    plt.title("Distribution of Standardised Residuals")
    plt.savefig(os.path.join(dataset_model_path, "residuals_distribution.png"), format='png', bbox_inches='tight')


    
    channel_stats = (
        pd.DataFrame(result.random_effects)
        .T
        .reset_index()
        .rename(columns={"index": "sub_ch"})
    )

    # Keep the slope column only; adjust the key to match your model term
    channel_stats["slope"] = channel_stats["C(state)[T.OT]"]

    plt.figure(figsize=(6,10))
    sns.pointplot(
        data=channel_stats.sort_values("slope"),
        y="sub_ch", x="slope", join=False, color="teal"
    )
    plt.axvline(0, color="k", lw=1)
    plt.xlabel("Random slope (state effect)")
    plt.ylabel("Sensor instance")
    plt.title("Caterpillar Plot of Random Slopes")
    plt.savefig(os.path.join(dataset_model_path, "caterpillar_plot_slopes.png"), format='png', bbox_inches='tight')

    sns.histplot(channel_stats["slope"], bins=30, kde=True)
    plt.axvline(channel_stats["slope"].mean(), color="k", lw=1)
    plt.xlabel("Random slope (state effect)")
    plt.title("Distribution of Channel Slopes")
    plt.savefig(os.path.join(dataset_model_path, "channel_slopes_distribution.png"), format='png', bbox_inches='tight')

    vc = pd.Series(result.cov_re.iloc[:,0], index=result.cov_re.index)
    vc.plot(kind="barh")
    plt.xlabel("Variance")
    plt.title("Estimated Random-Effect Variances")
    plt.savefig(os.path.join(dataset_model_path, "random_effect_variances.png"), format='png', bbox_inches='tight')
