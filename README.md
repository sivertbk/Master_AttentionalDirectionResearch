# Master Attentional Direction Research

## Project Overview
This project aims to analyze EEG data to study attentional direction in participants. The focus is on processing **Power Spectral Density (PSD)** data derived from EEG signals, using a structured and modular approach in Python.

The analysis workflow includes **preprocessing**, **frequency domain transformation**, and **statistical analysis**, with a focus on **alpha-band activity (8-12 Hz)** as a potential biomarker for attentional shifts.

---

## Project Structure
```
Master_AttentionalDirectionResearch/
│── data/                        # Raw and processed EEG data
│   ├── datasets/                 # Original EEG datasets
│   ├── epochs/                   # Preprocessed segmented epochs
│   ├── psd_data/                 # Power Spectral Density (PSD) data
│
│── eeg_analyzer/                 # Main module for EEG analysis
│   ├── __init__.py               # Module initialization
│   ├── eeg_analyzer.py           # EEGAnalyzer class (handles all subjects)
│   ├── metrics.py                # Computes EEG-specific metrics
│   ├── processor.py              # Processes PSD data (normalization, outlier removal)
│   ├── statistics.py             # Statistical modeling and hypothesis testing
│   ├── subject.py                # Manages individual subject data
│   ├── visualizer.py             # Visualization and plotting functions
│
│── logs/                         # Logging directory
│   ├── analysis_logs/            # Logs for analysis scripts
│   ├── preprocessing_logs/        # Logs for preprocessing steps
│
│── notebooks/                    # Jupyter Notebooks for experiments
│   ├── exploring.ipynb           # Exploratory data analysis
│   ├── hypothesis_testing.ipynb  # Hypothesis-driven analysis
│   ├── dataset_specific.ipynb    # One notebook per dataset (manual preprocessing)
│
│── plots/                        # Directory for generated plots
│   ├── psd_plots/                # Individual PSD visualizations
│   ├── subject_analysis/         # Subject-level comparisons
│   ├── summary_plots/            # Group-level insights
│
│── scripts/                      # Scripts for batch processing and analysis
│   ├── preprocess_data.py        # Filters and preprocesses EEG data
│   ├── compute_psd.py            # Converts EEG epochs to PSD data
│   ├── run_analysis.py           # Runs statistical analysis on PSD
│   ├── visualize_results.py      # Generates publication-ready plots
│
│── tests/                        # Unit tests for code validation
│   ├── __init__.py               # Module initialization
│   ├── test_metrics.py           # Tests for metrics calculations
│   ├── test_processing.py        # Tests for data processing logic
│
│── utils/                        # Utility functions and helpers
│   ├── __init__.py               # Module initialization
│   ├── file_io.py                # Functions for loading and saving data
│   ├── helpers.py                # Miscellaneous helper functions
│   ├── config.py                 # Global settings for file paths and parameters
│
│── README.md                     # Project documentation
│── requirements.txt               # Required Python dependencies
│── directory_tree.ipynb           # Notebook for project file structure
│── .gitignore                     # Files to exclude from version control
```

---

## Data Processing Workflow
The EEG data analysis follows these three major steps:

### 1. Preprocessing (Per Dataset, Requires Manual Steps)
- Each dataset has a dedicated **notebook** for preprocessing.
- Steps include:
  1. **Filtering** EEG data (high-pass filter at **1 Hz**).
  2. **Fixing/removing bad channels**.
  3. **Segmenting data into epochs**.
  4. **Removing bad epochs**.
  5. **Independent Component Analysis (ICA)** (manual selection of artifacts like eye blinks).
  6. **Exporting cleaned epochs** to `data/epochs/`.

Important: ICA selection requires manual input, so preprocessing is not fully automated.

---

### 2. Transforming Data to the Frequency Domain
- Converts EEG time-domain data into **Power Spectral Density (PSD)**.
- Output: PSD is stored as **nested dictionaries** for each subject.
- Optionally, **spectrograms** can be generated.

Output: PSD data is stored in `data/psd_data/` for analysis.

---

### 3. Processing and Analyzing PSD Data
This is where the core Python classes are used.

- **`Processor` (in `processor.py`)**
   - Handles normalization and outlier removal.
   - Works across all subjects and conditions.

- **`EEGAnalyzer` (in `eeg_analyzer.py`)**
   - Manages subject-level and group-level analysis.
   - Handles statistical modeling and comparisons.

Final Output: 
- Statistical results and visualizations of **how EEG spectral changes correlate with attentional direction**.

---

## Installation & Setup
### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/Master_AttentionalDirectionResearch.git
cd Master_AttentionalDirectionResearch
```

### 2. Create a Virtual Environment (Recommended)
```sh
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Running Jupyter Notebooks
```sh
jupyter notebook
```

---

## Configuration (`config.py`)
Global settings for the project are stored in `utils/config.py`. This includes:
- File paths for datasets, logs, and plots.
- Preprocessing settings (filtering, epoch length, ICA parameters).
- PSD settings (frequency resolution, normalization method).
- Outlier detection settings (z-score threshold).

Example usage:
```python
from utils.config import PSD_FREQUENCY_RES, PSD_NORMALIZATION
print(f"Using PSD frequency resolution: {PSD_FREQUENCY_RES} Hz")
```

---

## Next Steps
- Implement preprocessing notebooks for all datasets.
- Finalize PSD conversion scripts.
- Develop `Processor` class for data normalization.
- Validate and test the EEG analysis workflow.

---

## Contact & Contributions
If you're interested in contributing, feel free to fork the repository and submit a pull request.

