from eeg_analyzer.eeg_analyzer import EEGAnalyzer
import pandas as pd
from statsmodels.stats.multitest import multipletests
import os

def fit_mixedlm_unprocessed(analyzer: EEGAnalyzer):
    """
    Fits a separate linear mixed-effects model for each EEG channel to test
    the effect of attentional state on alpha-band power (in dB).
    Models include task orientation and dataset as fixed effects, and interactions.
    Applies FDR correction to the p-values for the main effect of state.

    Model formula: band_db ~ C(state) * C(task_orientation) * C(dataset)
    Random effects: Random intercepts for subject_session, random slopes for state per subject_session.
    """
    if analyzer is None:
        print("Analyzer not found. Create the analyzer first.")
        return
    
    if analyzer.df is None:
        print("Analyzer DataFrame is missing. Load or create the DataFrame first.")
        return

    # Ensure required columns exist in the DataFrame
    required_cols = ['band_db', 'state', 'task_orientation', 'dataset', 'subject_session', 'channel']
    missing_cols = [col for col in required_cols if col not in analyzer.df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in DataFrame: {missing_cols}. "
              "Ensure 'task_orientation' and other necessary columns are present.")
        return

    formula = "band_db ~ C(state) * C(task_orientation) * C(dataset)"
    value_col = 'band_db'  # Dependent variable, also used for data slicing checks
    state_col_for_analyzer_methods = 'state' # Primary state column for analyzer's internal processing
    group_col = "subject_session" # Grouping variable for random effects
    re_formula = "~C(state)"  # Random slope for state per subject_session (implies random intercept too)

    print(f"Starting per-channel mixed-effects model fitting with FDR correction.")
    print(f"Formula: {formula}")
    print(f"Random Effects: groups='{group_col}', re_formula='{re_formula}'")
    
    # Fit models using EEGAnalyzer's method. This populates analyzer.fitted_models.
    analyzer.fit_models_by_channel(
        formula=formula,
        value_col=value_col, 
        state_col=state_col_for_analyzer_methods,
        group_col=group_col,
        re_formula=re_formula,
        exclude_bad_rows=True # Assuming bad rows should be excluded
    )

    results_data = []
    p_values_for_fdr = []
    # Using a unique identifier (dataset + channel) for mapping FDR results back
    identifiers_for_fdr = [] 

    fitted_models_dict = analyzer.get_all_fitted_models()

    if not fitted_models_dict:
        print("No models were fitted by analyzer.fit_models_by_channel. Exiting analysis.")
        return

    print("Extracting p-values for the main effect of state from fitted models...")
    for dataset_name, models_in_dataset in fitted_models_dict.items():
        for model_identifier, model_result in models_in_dataset.items(): # model_identifier is channel_name
            channel_name = model_identifier 
            p_value_state = pd.NA # Use pandas NA for missing p-values
            converged = model_result.converged if model_result else False
            
            if model_result and converged:
                state_effect_term = None
                # Find the term representing the main effect of state (e.g., C(state)[T.MW])
                for term in model_result.pvalues.index:
                    if term.startswith("C(state)[T."): 
                        state_effect_term = term
                        break 
                
                if state_effect_term:
                    p_value_state = model_result.pvalues[state_effect_term]
                else:
                    # This might occur if 'state' has no variation in the data slice for a channel,
                    # or if the term is aliased due to collinearity.
                    print(f"Warning: Could not find main effect of state term (e.g., 'C(state)[T.*]') "
                          f"in p-values for {dataset_name}-{channel_name}. Model params: {list(model_result.pvalues.index)}")
            
            results_data.append({
                "dataset": dataset_name,
                "channel": channel_name,
                "p_value_state_uncorrected": p_value_state,
                "converged": converged
            })
            
            # Collect valid p-values for FDR correction
            if pd.notna(p_value_state):
                p_values_for_fdr.append(p_value_state)
                identifiers_for_fdr.append(f"{dataset_name}_{channel_name}")

    if not p_values_for_fdr:
        print("No valid p-values collected for the state effect. Cannot perform FDR correction.")
        if results_data: # Still print summary if some models were attempted but no p-values found
            summary_df = pd.DataFrame(results_data)
            print("\nSummary of Model Fitting (No FDR possible due to missing p-values):")
            print(summary_df.to_string())
        return

    # Apply FDR correction (Benjamini-Hochberg)
    reject, pvals_corrected, _, _ = multipletests(p_values_for_fdr, alpha=0.05, method='fdr_bh')

    # Map corrected p-values and significance back to the results_data list
    corrected_p_map = {id_key: pval for id_key, pval in zip(identifiers_for_fdr, pvals_corrected)}
    significance_map = {id_key: rej for id_key, rej in zip(identifiers_for_fdr, reject)}

    for row_dict in results_data:
        id_key = f"{row_dict['dataset']}_{row_dict['channel']}"
        if id_key in corrected_p_map: # Check if this channel's p-value was part of FDR
            row_dict["p_value_state_fdr_corrected"] = corrected_p_map[id_key]
            row_dict["significant_fdr"] = significance_map[id_key]
        else:
            row_dict["p_value_state_fdr_corrected"] = pd.NA
            row_dict["significant_fdr"] = pd.NA # Or False, if preferred for non-FDR-analyzed items

    summary_df = pd.DataFrame(results_data)
    
    # Define the desired column order for the output table
    output_columns_ordered = [
        "dataset", "channel", "converged", 
        "p_value_state_uncorrected", "p_value_state_fdr_corrected", "significant_fdr"
    ]
    # Filter to ensure only existing columns are selected, maintaining order
    final_summary_df = summary_df[[col for col in output_columns_ordered if col in summary_df.columns]]

    print("\n--- EEG Alpha Band Analysis: Per-Channel Mixed-Effects Models (FDR Corrected State Effect) ---")
    print(final_summary_df.to_string())

    # Save the summary table to a CSV file
    summary_filename = "alpha_state_effect_per_channel_fdr_summary.csv"
    if hasattr(analyzer, 'derivatives_path') and analyzer.derivatives_path:
        summary_filepath = os.path.join(analyzer.derivatives_path, summary_filename)
        try:
            final_summary_df.to_csv(summary_filepath, index=False)
            print(f"\nSummary table saved to: {summary_filepath}")
        except Exception as e:
            print(f"\nError saving summary table to {summary_filepath}: {e}")
    else:
        print("\nWarning: Analyzer derivatives_path not set or accessible. Cannot save summary table.")


if __name__ == "__main__":
    ANALYZER_NAME = "IQRFilteredSubExAnalyzer"

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print("Analyzer not found. Create the analyzer first.")
        exit()
    print(f"Loaded analyzer: {analyzer.analyzer_name}")
    if analyzer.df is None:
        print("Analyzer DataFrame is missing. Load or create the DataFrame first.")
        exit()

    # This function now performs the detailed per-channel analysis with FDR
    fit_mixedlm_unprocessed(analyzer)
    print("\nPer-channel mixed linear model fitting and FDR analysis completed.")

    # The following call summarizes ALL parameters of the fitted models, 
    # which is different from the FDR summary specific to the state effect.
    # It can be useful for a broader overview of all model coefficients.
    # Giving it a distinct filename to avoid confusion.
    analyzer.summarize_fitted_models(save=True, filename="all_fitted_models_full_summary.csv")
    print("General summary of all fitted model parameters generated.")

    analyzer.save_analyzer()  # Save the analyzer state, including .fitted_models
    print("Analyzer state saved.")

