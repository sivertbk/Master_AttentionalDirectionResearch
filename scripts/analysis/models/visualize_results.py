import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import PLOTS_PATH, set_plot_style

# Apply a consistent plot style
set_plot_style()


if __name__ == "__main__":
    ANALYZER_NAME = "UnprocessedAnalyzer" # Or any other analyzer with fitted models
    P_VALUE_THRESHOLD = 0.05

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer '{ANALYZER_NAME}' not found. Ensure it exists and has fitted models.")
        exit()
    print(f"Loaded analyzer: {analyzer.analyzer_name}")

    if analyzer.df is None: # Though df might not be strictly needed if model_result.model.data.frame is used
        print("Analyzer DataFrame is missing. This might be okay if model objects are self-contained.")
        # exit() # Not strictly an exit condition if model.data.frame is available

    all_fitted_models = analyzer.get_all_fitted_models()
    if not all_fitted_models:
        print(f"No fitted models found in analyzer '{ANALYZER_NAME}'.")
        exit()

    # Base save directory for all plots from this script run
    base_save_dir = os.path.join(PLOTS_PATH, "models", analyzer.analyzer_name, "significant_model_visualizations")
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"Saving plots to: {base_save_dir}")

    significant_models_found_count = 0

    for dataset_name, models_in_dataset in all_fitted_models.items():
        dataset_save_dir = os.path.join(base_save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)

        print(f"\nProcessing Dataset: {dataset_name}")

        for model_identifier, model_result in models_in_dataset.items():
            if model_result is None:
                # print(f"  Skipping {model_identifier}: Model result is None.")
                continue

            if not (hasattr(model_result, 'pvalues') and len(model_result.pvalues) > 1):
                # print(f"  Skipping {model_identifier}: Model has no pvalues or less than 2 parameters.")
                continue

            # Significance check (based on the p-value of the second parameter)
            slope_param_name = model_result.pvalues.index[1]
            slope_p_value = model_result.pvalues.iloc[1]

            if slope_p_value < P_VALUE_THRESHOLD:
                significant_models_found_count += 1
                print(f"  Significant model: {model_identifier} ({slope_param_name} p-value: {slope_p_value:.4f})")

                # Plot 1: Fitted values vs. Residuals
                try:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(model_result.fittedvalues, model_result.resid, alpha=0.5)
                    plt.axhline(0, color='red', linestyle='--')
                    plt.xlabel("Fitted Values")
                    plt.ylabel("Residuals")
                    plt.title(f"Fitted vs. Residuals\n{dataset_name} - {model_identifier}", fontsize=10)
                    plot_filename_resid = os.path.join(dataset_save_dir, f"{model_identifier}_fitted_vs_residuals.png")
                    plt.savefig(plot_filename_resid)
                    plt.close()
                    # print(f"    Saved: {plot_filename_resid}")
                except Exception as e:
                    print(f"    Error plotting fitted vs. residuals for {model_identifier}: {e}")

                # Plot 2: State Effect
                try:
                    # model_result.model.data.frame should contain the data used by patsy
                    if not hasattr(model_result.model, 'data') or not hasattr(model_result.model.data, 'frame'):
                        print(f"    Skipping state effect plot for {model_identifier}: model.data.frame not available.")
                        continue
                    
                    data_for_plot = model_result.model.data.frame.copy()
                    
                    value_col_plot = model_result.model.endog_names
                    if isinstance(value_col_plot, list): # Handle statsmodels >= 0.14
                         value_col_plot = value_col_plot[0]

                    state_col_plot = None
                    # Infer state column from exog names (e.g., 'C(state)[T.OT]')
                    for ex_name in model_result.model.exog_names:
                        if ex_name.startswith("C(") and "[T." in ex_name:
                            potential_col = ex_name.split('(')[1].split(')')[0]
                            if potential_col in data_for_plot.columns:
                                state_col_plot = potential_col
                                break
                    
                    if not state_col_plot: # Fallback if C() not used or different structure
                        if 'state' in data_for_plot.columns: # Common default
                            state_col_plot = 'state'
                        else: # Try to find a column that looks like a state from exog_names if it's simple
                            for ex_name in model_result.model.exog_names:
                                if ex_name in data_for_plot.columns and ex_name != 'Intercept':
                                    # Check if this column has few unique values, typical for a state
                                    if data_for_plot[ex_name].nunique() < 5 and data_for_plot[ex_name].nunique() > 1 : 
                                        state_col_plot = ex_name
                                        break
                    
                    if not state_col_plot:
                        print(f"    Could not reliably infer state column for {model_identifier}. Skipping state effect plot.")
                        continue

                    plt.figure(figsize=(8, 7))
                    sns.boxplot(x=state_col_plot, y=value_col_plot, data=data_for_plot, palette="Set2", showfliers=False)
                    sns.stripplot(x=state_col_plot, y=value_col_plot, data=data_for_plot, color='black', alpha=0.2, jitter=True, dodge=True, size=3)


                    # Overlay model's predicted means for each state
                    unique_states_in_data = sorted(data_for_plot[state_col_plot].unique())
                    predicted_means = {}

                    # Create a dataframe for prediction with one row per state, other variables at mean/mode
                    # This is more robust if there are other covariates.
                    # For simplicity, if only Intercept and state effect, we can calculate directly.
                    
                    # Simplified: assumes model like 'value ~ C(state)' or 'value ~ state'
                    if 'Intercept' in model_result.params.index:
                        ref_level_name = None
                        # Try to find the reference level if C() was used
                        for s_unique in unique_states_in_data:
                            is_ref = True
                            for ex_name in model_result.model.exog_names:
                                if f"C({state_col_plot})[T.{s_unique}]" == ex_name:
                                    is_ref = False
                                    break
                            if is_ref and f"C({state_col_plot})" in " ".join(model_result.model.exog_names): # Check if C() was used for state_col_plot
                                ref_level_name = s_unique
                                break
                        
                        if ref_level_name is None and len(unique_states_in_data) > 0 : # Fallback if not C() or only one level in exog_names
                            ref_level_name = unique_states_in_data[0]


                        for state_val in unique_states_in_data:
                            if state_val == ref_level_name:
                                predicted_means[state_val] = model_result.params['Intercept']
                            else:
                                state_param_effect_name = f"C({state_col_plot})[T.{state_val}]"
                                if state_param_effect_name in model_result.params.index:
                                    predicted_means[state_val] = model_result.params['Intercept'] + model_result.params[state_param_effect_name]
                                elif state_col_plot in model_result.params.index and len(unique_states_in_data) == 2: # For numeric state 0/1 or simple two-level factor
                                    # If state_col_plot is the second param (slope_param_name)
                                    if state_col_plot == slope_param_name:
                                        # This assumes state_val 1 corresponds to the effect, 0 to intercept
                                        # This part is heuristic and depends on how numeric states are coded
                                        if state_val == unique_states_in_data[1]: # Assuming second unique state has the effect
                                             predicted_means[state_val] = model_result.params['Intercept'] + model_result.params[slope_param_name]
                                        else:
                                             predicted_means[state_val] = model_result.params['Intercept']


                    if predicted_means:
                        # Plot means as horizontal lines on the boxplot
                        # Ensure x-coordinates match the boxplot categories
                        x_coords = {state: i for i, state in enumerate(unique_states_in_data)}
                        for state_val, mean_val in predicted_means.items():
                            if state_val in x_coords:
                                plt.hlines(mean_val, xmin=x_coords[state_val]-0.4, xmax=x_coords[state_val]+0.4, 
                                           colors='darkred', linestyles='--', lw=2.5, 
                                           label='Model Predicted Mean' if state_val == list(predicted_means.keys())[0] else "")
                        
                        handles, labels = plt.gca().get_legend_handles_labels()
                        if handles: # Add legend if there's something to label
                            by_label = dict(zip(labels, handles))
                            plt.legend(by_label.values(), by_label.keys(), loc='best')
                    
                    plt.xlabel(f"State ({state_col_plot})")
                    plt.ylabel(f"{value_col_plot}")
                    plt.title(f"Effect of State on {value_col_plot}\n{dataset_name} - {model_identifier} (p={slope_p_value:.4f})", fontsize=10)
                    plt.tight_layout()
                    plot_filename_state = os.path.join(dataset_save_dir, f"{model_identifier}_state_effect.png")
                    plt.savefig(plot_filename_state)
                    plt.close()
                    # print(f"    Saved: {plot_filename_state}")

                except Exception as e:
                    print(f"    Error plotting state effect for {model_identifier}: {e}")
            # else:
                # print(f"  Skipping {model_identifier}: Not significant (p-value: {slope_p_value:.4f}).")
    
    if significant_models_found_count == 0:
        print(f"\nNo significant models found with p-value < {P_VALUE_THRESHOLD} to visualize.")
    else:
        print(f"\nFinished visualizing {significant_models_found_count} significant models.")

    print("Visualization script finished.")





