# 🧠 EEG Preprocessing Pipeline

This pipeline is designed to **maximize signal integrity** for **spectral and time-frequency EEG analyses**. It avoids distortive filtering on the final data while still removing artifacts and noise through robust preprocessing techniques like RANSAC, AutoReject, and ICA.

---

## 🧩 Step-by-Step Pipeline Overview

### 1. Load and Downsample Raw EEG Data
Load subject data, set proper channel types, and downsample using the target sampling rate defined in `config.py`.

### 2. Detect Bad Channels Using RANSAC
- Create a **0.1 Hz high-passed copy** of the raw data.
- Generate **fixed-length synthetic epochs**.
- Fit a `RANSAC` object on the epochs to identify consistently noisy channels.

### 3. Interpolate Bad Channels in the Raw
- Mark `ransac.bad_chs_` in the original raw object.
- Interpolate them directly in the **unfiltered raw data**.

### 4. Create Synthetic Epochs with Detrending
- From the interpolated raw, generate **new fixed-length epochs**.
- Apply **linear detrending** (`detrend=1`) to remove drift **without filtering**.

### 5. Apply AutoReject to Detect Bad Epochs
- Use AutoReject to automatically reject epochs with sharp transients or large deviations.
- This improves data quality for ICA fitting.

### 6. Apply Average Reference
- Apply **average referencing** only after removing bad epochs.
- This prevents bad channels from biasing the reference signal.

### 7. Fit ICA on Clean Epochs
- Fit ICA only on **AutoReject-approved, referenced epochs**.
- Use `n_components=0.99` to retain 99% of the variance while avoiding overfitting.

### 8. Automatically Identify EOG Components
- Use correlation with EOG channels to detect and flag eye-related ICA components.
- Update `ica.exclude` accordingly.

### 9. Apply ICA to Full Unfiltered Raw
- Apply the ICA model to the **full unfiltered raw object**.
- This ensures cleaned data with intact low-frequency information.

### 10. Create Task Epochs and Apply Detrending
- Epoch the ICA-cleaned raw based on experimental events.
- Use `detrend=1` to remove slow drifts **without altering the frequency spectrum**.

### 11. Apply Final AutoReject to Task Epochs
- Use AutoReject again on the final epochs to remove remaining transients or edge effects.

### 12. Save Final Outputs and Logs
- Save:
  - `epochs_final.fif`
  - ICA model and excluded components
  - `bad_channels.json` or `.csv`
  - Reject logs and diagnostic figures
  - Any plots from ICA or AutoReject (e.g., topographies, component activations)

---

## 🧠 Why This Pipeline Works for Frequency Analysis

This pipeline avoids filtering the final data to ensure **spectral integrity** — ideal for power spectral density (PSD), time-frequency, or connectivity analyses.  
Key decisions include:

- **Filtering only helper copies** used for detecting noise and fitting ICA.
- Using **linear detrending** (instead of high-pass filters) on the final epochs.
- Applying ICA trained on clean surrogate data to the full raw to avoid overfitting.
- Ensuring that referencing and interpolation are done **after** artifact detection, not before.

The result: **maximally artifact-free**, **minimally distorted**, and **spectrally faithful** EEG epochs for high-quality frequency-domain analysis.

---
