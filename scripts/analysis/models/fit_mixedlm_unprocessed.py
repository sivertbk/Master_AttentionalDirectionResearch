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

    # Also fit models by ROI using the same parameters
    print(f"Starting per-ROI mixed-effects model fitting with same formula.")
    analyzer.fit_models_by_roi(
        formula=formula,
        value_col=value_col,
        state_col=state_col_for_analyzer_methods,
        group_col=group_col,
        re_formula=re_formula,
        exclude_bad_rows=True
    )

    results_data = []
    # Collect p-values organized by dataset and identifier type (channel vs ROI)
    dataset_type_p_values = {}  # dataset_name -> {channels: {p_values: [...], identifiers: [...]}, rois: {p_values: [...], identifiers: [...]}}

    fitted_models_dict = analyzer.get_all_fitted_models()

    if not fitted_models_dict:
        print("No models were fitted by analyzer.fit_models_by_channel or fit_models_by_roi. Exiting analysis.")
        return

    print("Extracting p-values for the main effect of state from fitted models (channels and ROIs)...")
    for dataset_name, models_in_dataset in fitted_models_dict.items():
        # Initialize dataset entry if not exists
        if dataset_name not in dataset_type_p_values:
            dataset_type_p_values[dataset_name] = {
                "channels": {"p_values": [], "identifiers": []},
                "rois": {"p_values": [], "identifiers": []}
            }
        
        for model_identifier, model_result in models_in_dataset.items(): # model_identifier is channel_name or roi_identifier
            identifier_name = model_identifier 
            # Determine if this is a channel or ROI based on identifier format
            identifier_type = "ROI" if "_" in model_identifier and any(region in model_identifier for region in ["frontal", "parietal", "central", "temporal", "occipital", "prefrontal", "frontocentral", "centroparietal", "parietooccipital", "fronto-parietal"]) else "channel"
            
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
                    # This might occur if 'state' has no variation in the data slice for a channel/ROI,
                    # or if the term is aliased due to collinearity.
                    print(f"Warning: Could not find main effect of state term (e.g., 'C(state)[T.*]') "
                          f"in p-values for {dataset_name}-{identifier_name}. Model params: {list(model_result.pvalues.index)}")
            
            results_data.append({
                "dataset": dataset_name,
                "identifier": identifier_name,
                "identifier_type": identifier_type,
                "p_value_state_uncorrected": p_value_state,
                "converged": converged
            })
            
            # Collect valid p-values for FDR correction, organized by dataset and type
            if pd.notna(p_value_state):
                identifier_key = f"{dataset_name}_{identifier_name}"
                if identifier_type == "channel":
                    dataset_type_p_values[dataset_name]["channels"]["p_values"].append(p_value_state)
                    dataset_type_p_values[dataset_name]["channels"]["identifiers"].append(identifier_key)
                else:  # ROI
                    dataset_type_p_values[dataset_name]["rois"]["p_values"].append(p_value_state)
                    dataset_type_p_values[dataset_name]["rois"]["identifiers"].append(identifier_key)

    # Apply FDR correction within each dataset and type separately
    all_corrected_p_map = {}
    all_significance_map = {}
    
    for dataset_name, dataset_data in dataset_type_p_values.items():
        # Process channels for this dataset
        channel_p_vals = dataset_data["channels"]["p_values"]
        channel_identifiers = dataset_data["channels"]["identifiers"]
        
        if channel_p_vals:
            print(f"Applying FDR correction to {len(channel_p_vals)} channel p-values in dataset '{dataset_name}'...")
            reject, pvals_corrected, _, _ = multipletests(channel_p_vals, alpha=0.05, method='fdr_bh')
            
            # Store corrected p-values and significance for channels in this dataset
            for id_key, pval_corr, is_sig in zip(channel_identifiers, pvals_corrected, reject):
                all_corrected_p_map[id_key] = pval_corr
                all_significance_map[id_key] = is_sig
        else:
            print(f"No valid channel p-values found for dataset '{dataset_name}' - skipping channel FDR correction.")
        
        # Process ROIs for this dataset
        roi_p_vals = dataset_data["rois"]["p_values"]
        roi_identifiers = dataset_data["rois"]["identifiers"]
        
        if roi_p_vals:
            print(f"Applying FDR correction to {len(roi_p_vals)} ROI p-values in dataset '{dataset_name}'...")
            reject, pvals_corrected, _, _ = multipletests(roi_p_vals, alpha=0.05, method='fdr_bh')
            
            # Store corrected p-values and significance for ROIs in this dataset
            for id_key, pval_corr, is_sig in zip(roi_identifiers, pvals_corrected, reject):
                all_corrected_p_map[id_key] = pval_corr
                all_significance_map[id_key] = is_sig
        else:
            print(f"No valid ROI p-values found for dataset '{dataset_name}' - skipping ROI FDR correction.")

    # Check if any corrections were applied
    total_corrections = sum(
        len(data["channels"]["p_values"]) + len(data["rois"]["p_values"]) 
        for data in dataset_type_p_values.values()
    )
    if total_corrections == 0:
        print("No valid p-values collected for the state effect. Cannot perform FDR correction.")
        if results_data: # Still print summary if some models were attempted but no p-values found
            summary_df = pd.DataFrame(results_data)
            print("\nSummary of Model Fitting (No FDR possible due to missing p-values):")
            print(summary_df.to_string())
        return

    # Map corrected p-values and significance back to the results_data list
    for row_dict in results_data:
        id_key = f"{row_dict['dataset']}_{row_dict['identifier']}"
        if id_key in all_corrected_p_map:
            row_dict["p_value_state_fdr_corrected"] = all_corrected_p_map[id_key]
            row_dict["significant_fdr"] = all_significance_map[id_key]
        else:
            row_dict["p_value_state_fdr_corrected"] = pd.NA
            row_dict["significant_fdr"] = pd.NA

    summary_df = pd.DataFrame(results_data)
    
    # Define the desired column order for the output table
    output_columns_ordered = [
        "dataset", "identifier", "identifier_type", "converged", 
        "p_value_state_uncorrected", "p_value_state_fdr_corrected", "significant_fdr"
    ]
    # Filter to ensure only existing columns are selected, maintaining order
    final_summary_df = summary_df[[col for col in output_columns_ordered if col in summary_df.columns]]

    print("\n--- EEG Alpha Band Analysis: Per-Channel and Per-ROI Mixed-Effects Models (FDR by Dataset and Type) ---")
    for dataset_name, dataset_data in dataset_type_p_values.items():
        n_channels = len(dataset_data['channels']['p_values'])
        n_rois = len(dataset_data['rois']['p_values'])
        print(f"Dataset '{dataset_name}': {n_channels} channels, {n_rois} ROIs corrected separately")
    print(final_summary_df.to_string())

    # Save the summary table to a CSV file
    summary_filename = "alpha_state_effect_per_channel_and_roi_fdr_summary.csv"
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
    ANALYZER_NAME = "UnprocessedAnalyzer"

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print("Analyzer not found. Create the analyzer first.")
        exit()
    print(f"Loaded analyzer: {analyzer.analyzer_name}")
    if analyzer.df is None:
        print("Analyzer DataFrame is missing. Load or create the DataFrame first.")
        exit()

    # This function now performs the detailed per-channel and per-ROI analysis with FDR
    fit_mixedlm_unprocessed(analyzer)
    print("\nPer-channel and per-ROI mixed linear model fitting and FDR analysis completed.")

    # The following call summarizes ALL parameters of the fitted models, 
    # which is different from the FDR summary specific to the state effect.
    # It can be useful for a broader overview of all model coefficients.
    # Giving it a distinct filename to avoid confusion.
    analyzer.summarize_fitted_models(save=True, filename="all_fitted_models_full_summary.csv")
    print("General summary of all fitted model parameters generated.")

    analyzer.save_analyzer()  # Save the analyzer state, including .fitted_models
    print("Analyzer state saved.")

