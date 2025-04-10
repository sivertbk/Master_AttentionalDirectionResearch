# Presentation Overview: EEG Preprocessing Pipeline and Analysis Proposal

---

## 1. Repository & Code Organization

- **Structured Repository:**  
  - Reorganized the codebase into a clear and structured repository.
  - Comprehensive configuration file supports multiple datasets.
  - Code has been cleaned and modularized using object-oriented practices.
  
- **Tracking & Logging:**  
  - Each processing step logs critical outputs (e.g., bad channels, AutoReject logs, ICA components).
  - All processing reports, figures, and intermediate outputs are saved to dedicated folders (JSON/CSV for metadata, plots for visual checks).

---

## 2. PSD Computation Script

- **What It Does:**  
  - Computes Power Spectral Density (PSD) for each subject.
  - Generates multi-dimensional arrays with dimensions `(n_epochs, n_channels, n_freqs)`.
  - Associates dynamic epoch metadata (task, state, etc.) and static subject-specific parameters (e.g., Welch method settings).
  
- **Key Parameter Choices:**  
  - Utilizes the Welch method with a tuned parameter set:
    - Optimized to sacrifice some spectral resolution for reduced variance.
    - This boosts the statistical power for later analyses.

---

## 3. Current Pipeline (Data Integrity Focus)

- **Core Principle:**  
  - Preserve the original spectral content for frequency analysis.
  - Filter only the copies used to derive helper epochs (for artifact detection) while keeping the final epochs unfiltered.

- **Pipeline Outline:**  
  1. **Load and Downsample Raw EEG Data:**  
     - Load raw data.
     - Downsample as defined in the configuration.

  2. **Detect Bad Channels Using RANSAC:**  
     - Create a high-pass (0.1 Hz) filtered copy.
     - Generate fixed-length synthetic epochs.
     - Use RANSAC to detect consistently bad channels.
  
  3. **Interpolate Bad Channels:**  
     - Apply interpolation on the original unfiltered raw data using RANSAC results.
  
  4. **Create Synthetic Epochs for Artifact Detection:**  
     - Use the interpolated raw data.
     - Create fixed-length epochs with **linear detrending** (no high-pass filter) to maintain spectral properties.
  
  5. **Run AutoReject:**  
     - Apply AutoReject on these synthetic epochs to remove epochs with sharp transients.
  
  6. **Average Reference:**  
     - Apply average referencing after cleaning so that the reference is not skewed by noisy channels.
  
  7. **Fit ICA:**  
     - Fit ICA on the clean, average-referenced epochs (using only good epochs).
     - Use `n_components=0.99` to capture 99% of data variance.
  
  8. **Identify and Exclude EOG Components:**  
     - Automatically flag eye artifacts using dedicated EOG channels.
  
  9. **Apply ICA to Full Unfiltered Raw Data:**  
     - Clean the full raw (unfiltered) data using the ICA solution.
  
  10. **Epoch for Analysis:**  
      - Create analysis-specific epochs (e.g., 5-second pre-event epochs) from the ICA-cleaned raw.
      - Apply detrending to remove slow drifts.
  
  11. **Final Cleaning with AutoReject:**  
      - Run AutoReject again on the final epochs to remove any residual artifacts.
  
  12. **Save All Outputs:**  
      - Save final epochs, ICA details, bad channel lists, and logs.

---

## 4. New Dataset Proposal: Driving Task

- **Motivation:**  
  - The current external task data (a boring task designed to induce mind wandering) may not fully span the attentional direction continuum.
  
- **Characteristics of the Driving Task Dataset:**  
  - **Continuous Motor & Sensory Engagement:** Involves active driving with realistic cognitive demands.
  - **Dynamic Environmental Stimuli:** Offers naturalistic perturbations and continuous task demands.
  
- **Why It’s Valuable:**  
  - Provides a **richer testbed** for understanding externally directed attention.
  - Likely yields a **larger FOCUS–MW alpha difference** due to increased task engagement compared to simpler tasks (e.g., SART, visual search).

---

## 5. Labeling Method for the Driving Task

- **Proposed Approach Without Probes:**  
  - **Derive MW and FOCUS segments using event-based behavioral proxies:**
  
  **Likely Mind-Wandering (MW) Segments:**  
  - Missed police car detection (event `2811` without a subsequent `4621` within 5s).
  - Vehicle collisions (`4421`).
  - False alarms (`4720`).

  **Likely Focused Attention (FOCUS) Segments:**  
  - Accurate and timely police car detection (`4621` within 5s of `2811`).
  - 5 seconds post-collision (indicative of post-error vigilance).

- **Outcome:**  
  - Provides reasonably labeled segments for further analysis of attentional state.

---

## 6. Analysis Strategy & Hypotheses

- **Overall Aim:**  
  - Compare total **alpha-band (8–12 Hz)** power between FOCUS and Mind-Wandering states within subjects and across datasets.
  
- **Key Analytical Considerations:**  
  - **Within-Subject Comparisons:** Use paired analyses of alpha power differences.
  - **Inter-Subject Variability:** Employ Linear Mixed-Effects Models (LMMs) with random intercepts and slopes.
  - **Task Context:** Include task type (internal vs. external) as a high-level factor.
  
- **Statistical Methods:**  
  - Non-parametric testing (Mann-Whitney U, permutation cluster tests).
  - LMM framework to extract slopes representing changes between conditions.

- **Hypothesis:**  
  - Expect a **larger FOCUS–MW alpha difference in the driving task** relative to other tasks.  
  - This would support that attentional direction is reliably reflected in
