"""
Step 3: ICA Artifact Detection Pipeline

This script orchestrates the complete ICA workflow by running two processes in sequence:
1. First: Fit ICA models on cleaned epochs
2. Second: Detect and save EOG artifacts (blinks and saccades) using the fitted ICA models

The script can be run standalone or the individual functions can be imported and used elsewhere.
"""

import time
from utils.config import DATASETS

# Import the wrapped functions
from .fit_ica import fit_ica_models
from .ica_artifact_detection import detect_ica_artifacts


def run_step3_ica_pipeline(datasets=None, verbose=True, show_plots=False, 
                          z_threshold_blink=5.0, z_threshold_saccade=5.0):
    """
    Run the complete Step 3 ICA pipeline: fitting ICA models and detecting artifacts.
    
    Parameters
    ----------
    datasets : dict, optional
        Dictionary of datasets to process. If None, uses DATASETS from config.
    verbose : bool, optional
        Whether to print verbose output. Default is True.
    show_plots : bool, optional
        Whether to show plots during processing. Default is False.
    z_threshold_blink : float, optional
        Z-score threshold for blink detection. Default is 5.0.
    z_threshold_saccade : float, optional
        Z-score threshold for saccade detection. Default is 5.0.
    
    Returns
    -------
    dict
        Summary of processing results including counts for each step.
    """
    print("="*80)
    print("           STEP 3: ICA ARTIFACT DETECTION PIPELINE")
    print("="*80)
    
    start_time = time.time()
    
    if datasets is None:
        datasets = DATASETS.copy()
        # Configure datasets for ICA processing
        # datasets.pop('braboszcz2017', None)  # Remove if not ready
        # datasets.pop('jin2019', None)       # Remove if not ready  
        # datasets.pop('touryan2022', None)   # Remove if not ready
    
    print(f"[INFO] Processing datasets: {list(datasets.keys())}")
    print(f"[INFO] Verbose mode: {verbose}")
    print(f"[INFO] Show plots: {show_plots}")
    print(f"[INFO] Blink detection threshold: {z_threshold_blink}")
    print(f"[INFO] Saccade detection threshold: {z_threshold_saccade}")
    print("-"*80)
    
    # Step 3.1: Fit ICA Models
    print("\n[STEP 3.1] Starting ICA model fitting...")
    step1_start = time.time()
    
    ica_fit_results = fit_ica_models(
        datasets=datasets,
        verbose=verbose,
        show_plots=show_plots
    )
    
    step1_time = time.time() - step1_start
    print(f"[STEP 3.1] ICA fitting completed in {step1_time:.1f} seconds")
    print(f"[STEP 3.1] Results: {ica_fit_results}")
    
    # Step 3.2: Detect ICA Artifacts
    print("\n[STEP 3.2] Starting ICA artifact detection...")
    step2_start = time.time()
    
    artifact_detection_results = detect_ica_artifacts(
        datasets=datasets,
        verbose=verbose,
        show_plots=show_plots,
        z_threshold_blink=z_threshold_blink,
        z_threshold_saccade=z_threshold_saccade
    )
    
    step2_time = time.time() - step2_start
    print(f"[STEP 3.2] Artifact detection completed in {step2_time:.1f} seconds")
    print(f"[STEP 3.2] Results: {artifact_detection_results}")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("                    STEP 3 PIPELINE SUMMARY")
    print("="*80)
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"ICA fitting time: {step1_time:.1f} seconds")
    print(f"Artifact detection time: {step2_time:.1f} seconds")
    print(f"Total subjects processed (ICA): {ica_fit_results.get('processed', 0)}")
    print(f"Total subjects processed (Artifacts): {artifact_detection_results.get('processed', 0)}")
    print(f"Total subjects skipped (ICA): {ica_fit_results.get('skipped', 0)}")
    print(f"Total subjects skipped (Artifacts): {artifact_detection_results.get('skipped', 0)}")
    print("="*80)
    
    return {
        "total_time": total_time,
        "ica_fit_results": ica_fit_results,
        "artifact_detection_results": artifact_detection_results,
        "step1_time": step1_time,
        "step2_time": step2_time
    }


if __name__ == "__main__":
    # Run the complete pipeline when script is executed directly
    results = run_step3_ica_pipeline(
        verbose=True,
        show_plots=False,
        z_threshold_blink=5.0,
        z_threshold_saccade=5.0
    )
    
    print(f"\n[INFO] Step 3 pipeline completed successfully!")
    print(f"[INFO] Check the results dictionary for detailed information.")
