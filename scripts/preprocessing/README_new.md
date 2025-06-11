# EEG Preprocessing Pipeline

This document outlines the step-by-step preprocessing pipeline for EEG data, designed to clean raw recordings and prepare them for spectral analysis, such as Power Spectral Density (PSD) estimation. The pipeline emphasizes robust artifact removal while preserving data integrity.

**It is crucial to visually inspect all generated figures at each step to ensure the quality and validity of the preprocessing.**

## 🧩 Pipeline Overview

The preprocessing workflow consists of the following main stages, executed by a series of Python scripts:

1.  **Bad Channel Detection (RANSAC)**: Identifies noisy or malfunctioning channels.
2.  **Manual Inspection and Verification of Bad Channels**: Allows for review and correction of automatically detected bad channels.
3.  **ICA Model Fitting**: Trains Independent Component Analysis (ICA) models to separate neural signals from artifacts.
4.  **ICA-based Artifact Detection**: Identifies ICA components corresponding to biological artifacts (e.g., eye blinks, saccades).
5.  **Main Preprocessing**: Applies corrections and transformations to the raw data to produce analysis-ready epochs.
6.  **Inspection of Analysis-Ready Epochs**: Final visual check of the cleaned epochs.

---

## ⚙️ Step-by-Step Script Execution and Details

### 1. Detect Bad Channels (`detect_bad_channels.py`)

*   **Purpose**: Automatically identify and flag bad EEG channels using the RANSAC (Random Sample Consensus) algorithm.
*   **Process**:
    *   Loads raw EEG data for each subject and dataset.
    *   Prepares the raw data by setting channel types and downsampling according to `EEG_SETTINGS` in `config.py`.
    *   Applies RANSAC to a high-passed (0.1 Hz) copy of the data, using fixed-length synthetic epochs to detect channels that are consistently noisy.
*   **Output**:
    *   A JSON file for each subject/session, listing the channels identified as bad by RANSAC.
    *   Plots visualizing the RANSAC procedure and detected bad channels (e.g., channel-wise variance).
*   **Action Required**: Inspect the generated plots to get an initial understanding of channel quality.

### 2. Inspect and Verify Bad Channels (`inspect_bad_channels.py`)

*   **Purpose**: Manually review and verify the bad channels detected by RANSAC. This step allows for correction of any false positives or omissions from the automated detection.
*   **Process**:
    *   Loads the raw EEG data and the RANSAC-detected bad channels list (JSON file from the previous step).
    *   Marks these channels as 'bad' in the MNE-Python `raw` object.
    *   Plots the raw data with the marked bad channels, allowing for visual inspection. The user can interactively add or remove channels from the 'bads' list within the MNE plot window.
*   **Output**:
    *   An updated JSON file for each subject/session, named with a `_verified` suffix (or similar, check `update_bad_channels_json` in `file_io.py` for exact naming), containing the manually confirmed list of bad channels. This file will be used in the main preprocessing script for interpolation.
    *   Topographical plots showing the final set of interpolated bad channels.
*   **Action Required**: Carefully inspect the raw data plots. Add or remove channels from the `raw.info['bads']` list as needed before closing the plot window to save the verified list.

### 3. Fit ICA Models (`fit_ica.py`)

*   **Purpose**: Train an ICA model for each subject/session to decompose the EEG signal into statistically independent components. This is a preparatory step for artifact removal.
*   **Process**:
    *   Loads the raw EEG data.
    *   Creates synthetic epochs specifically for ICA fitting. These epochs are typically filtered (e.g., 1 Hz high-pass) and may undergo automated epoch rejection (e.g., using AutoReject or a simple peak-to-peak threshold) to ensure the ICA model is trained on relatively clean data. The script uses `prepare_ica_epochs` which involves creating fixed-length epochs and applying rejection based on `min_threshold` and `reject_scale_factor`.
    *   Fits an ICA model (e.g., using the `infomax` or `picard` algorithm) to these prepared epochs. The number of components can be set to retain a certain percentage of variance (e.g., 99% as mentioned in the initial README concept) or a fixed number.
*   **Output**:
    *   An MNE-Python ICA object (`.fif` file) saved for each subject/session, containing the fitted ICA solution.
    *   The epochs used for ICA fitting are also saved.
*   **Action Required**: No direct action, but be aware that the quality of ICA decomposition depends on the quality of the data fed into it.

### 4. ICA-based Artifact Detection (`ica_artifact_detection.py`)

*   **Purpose**: Automatically identify ICA components that capture common biological artifacts, primarily eye blinks and saccades, by correlating component activity with EOG channel data.
*   **Process**:
    *   Loads the epochs used for ICA fitting (from `fit_ica.py`) and the corresponding fitted ICA model.
    *   Creates bipolar EOG channels (VEOG, HEOG) from the original EOG channels (e.g., UVEOG, LVEOG, LHEOG, RHEOG).
    *   Uses MNE-Python's `ica.find_bads_eog()` method to find components highly correlated with the VEOG (for blinks) and HEOG (for saccades) channels. This method typically uses a z-score threshold to identify artifactual components. The script uses `z_threshold_blink` and `z_threshold_saccade`.
    *   Generates various plots to help assess the identified components:
        *   Score plots showing the correlation/z-score of each component with the EOG channels.
        *   ICA component property plots (topography, activation, PSD, variance) for the identified blink and saccade components.
        *   EOG evoked responses before and after applying ICA with the detected artifactual components excluded (to verify cleaning).
        *   EOG overlay plots.
*   **Output**:
    *   A JSON file for each subject/session, listing the indices of the ICA components identified as EOG-related artifacts (blinks and saccades).
    *   A comprehensive set of plots saved to the `reports/plots/` directory, organized by dataset and plot type (e.g., `ica_scores_blinks`, `ica_properties`, `eog_evoked_before_after`).
*   **Action Required**: **Crucially, inspect all generated plots.**
    *   Verify that the components marked for exclusion indeed represent eye artifacts (typical frontally-dominant topographies, characteristic time courses).
    *   Ensure that the EOG evoked responses show a clear reduction of artifactual activity after ICA.
    *   If the automatic detection is unsatisfactory, manual adjustment of the `ica.exclude` list might be necessary, though this script focuses on automatic detection and saving those results.

### 5. Main Preprocessing (`main_prep.py`)

*   **Purpose**: Apply all determined corrections (bad channel interpolation, ICA artifact removal) and prepare the data into analysis-ready epochs.
*   **Process**:
    1.  **Load and Prepare Raw Data**:
        *   Loads the original raw EEG data.
        *   Applies initial preparation steps like downsampling and setting channel types (via `prepare_raw_data`).
    2.  **Interpolate Bad Channels**:
        *   Loads the **verified** list of bad channels (JSON from `inspect_bad_channels.py`).
        *   Interpolates these bad channels in the raw data (via `fix_bad_channels`).
    3.  **Apply ICA**:
        *   Applies a common average reference to the data, as ICA models were likely trained on referenced data.
        *   Loads the fitted ICA model (from `fit_ica.py`) and the list of excluded EOG components (JSON from `ica_artifact_detection.py`).
        *   Applies the ICA solution to the raw data, removing the artifactual components.
        *   Optionally creates and plots EOG evoked responses from the raw data before and after ICA to verify cleaning on the continuous data (if `PLOT_EOG_RAW` is `True`).
    4.  **Epoch Data**:
        *   Segments the ICA-cleaned raw data into epochs based on experimental events/conditions defined in the dataset configuration (via `create_analysis_epochs`).
        *   Applies linear detrending to each epoch to remove slow drifts.
        *   Applies baseline correction.
    5.  **Automated Epoch Rejection**:
        *   Concatenates epochs from all conditions for the current subject/session.
        *   Calculates a subject-specific peak-to-peak amplitude rejection threshold using a Bayesian optimization method (via `get_scaled_rejection_threshold`). This threshold is logged.
        *   Applies this threshold to reject epochs containing excessive noise. The `epochs_dict` (epochs per condition) is then processed to drop these bad epochs.
    6.  **Save Outputs and Quality Control**:
        *   Saves the final cleaned epochs for each experimental condition to disk (`.fif` files).
        *   Logs the number of dropped epochs and the rejection thresholds used.
        *   Generates and saves quality control plots:
            *   Comparison of Power Spectral Density (PSD) before and after epoch rejection.
            *   Comparison of evoked responses before and after epoch rejection.
*   **Output**:
    *   Analysis-ready, cleaned epoch files (`_epo.fif`) for each subject, session, and condition, saved in the `data/<dataset>/epochs/analysis_epochs/` directory.
    *   Log files detailing rejection thresholds and dropped epoch counts.
    *   QC plots (PSDs, evoked responses) saved in `reports/plots/`.
*   **Action Required**:
    *   Inspect the QC plots (PSDs and evoked responses) to ensure that epoch rejection did not inadvertently remove critical data or introduce biases.
    *   Check the logs for the number of dropped epochs; an excessively high number might indicate issues earlier in the pipeline or very noisy data.

### 6. Inspect Analysis-Ready Epochs (`inspect_analysis_epochs.py`)

*   **Purpose**: Provide a final opportunity to visually inspect the cleaned, analysis-ready epochs.
*   **Process**:
    *   Loads the saved analysis epochs for a specified dataset, subject, and session (or iterates through all).
    *   Plots the epochs for interactive inspection (scrolling, marking bad).
    *   Plots PSD and PSD topomaps for the epochs.
    *   If any epochs are interactively marked as bad during inspection, the script can update the epoch file and log the changes.
*   **Output**:
    *   Potentially updated epoch files if manual changes are made during inspection.
    *   Updated logs of dropped epochs.
*   **Action Required**: Visually inspect the epochs. Check for any remaining artifacts or unusual patterns. This is the last chance to catch issues before proceeding to spectral analysis.

---

## Expected Results

Upon successful completion of this pipeline, you will have:

*   **Cleaned EEG Epochs**: For each subject and experimental condition, a set of `.fif` files containing preprocessed epochs, ready for frequency analysis (e.g., PSD estimation, time-frequency analysis). These are stored in `data/<dataset_name>/epochs/analysis_epochs/`.
*   **ICA Models and Excluded Components**: Saved ICA solutions and lists of artifactual components for reproducibility and review.
*   **Bad Channel Lists**: JSON files documenting automatically detected and manually verified bad channels.
*   **Logs**: Records of epoch rejection thresholds and counts of dropped epochs.
*   **Diagnostic Plots**: A comprehensive collection of plots (channel quality, ICA component properties, EOG cleaning verification, PSDs, evoked responses) stored in `reports/plots/` that document the preprocessing steps and allow for quality assessment.

This pipeline aims to maximize signal integrity by avoiding distortive filtering on the final data, relying instead on robust techniques like RANSAC, AutoReject (implicitly via thresholding in `main_prep.py`), and ICA for artifact and noise removal.
