# Master Attentional Direction Research

## Project Overview
This project aims to analyze EEG data to study attentional direction in participants. The focus is on processing **Power Spectral Density (PSD)** data derived from EEG signals, using a structured and modular approach in Python.

The analysis workflow includes **preprocessing**, **frequency domain transformation**, and **statistical analysis**, with a focus on **alpha-band activity (8-12 Hz)** as a potential biomarker for attentional shifts.

---

## Project Structure
```
attentional_direction_research_workspace/
│── backups/                        # Backup files for analysis
│
│── data/                           # EEG datasets and processed data
│   ├── datasets/                   # Original EEG datasets
│   ├── epochs/                     # Preprocessed segmented epochs
│   ├── psd_data/                   # Power Spectral Density (PSD) data
│
│── eeg_analyzer/                    # Main module for EEG analysis
│   ├── __init__.py                  # Module initialization
│   ├── eeg_analyzer.py              # EEGAnalyzer class (handles all subjects)
│   ├── metrics.py                   # Computes EEG-specific metrics
│   ├── processor.py                 # Processes PSD data (normalization, outlier removal)
│   ├── statistics.py                # Statistical modeling and hypothesis testing
│   ├── subject.py                   # Manages individual subject data
│   ├── visualizer.py                 # Visualization and plotting functions
│
│── notebooks/                       # Jupyter Notebooks for experiments
│   ├── exploring.ipynb              # Exploratory data analysis
│   ├── hypothesis_testing.ipynb     # Hypothesis-driven analysis
│
│── reports/                         # Generated reports, logs, and plots
│   ├── logs/                        # Logs for analysis and preprocessing
│   │   ├── analysis_logs/           # Logs for EEG analysis scripts
│   │   ├── preprocessing_logs/      # Logs for preprocessing steps
│   ├── plots/                       # Directory for generated plots
│
│── scripts/                         # Standalone scripts for batch processing and analysis
│   ├── dir_tree.py                  # Script for generating directory tree
│
│── tests/                           # Unit tests for validating code
│   ├── __init__.py                  # Module initialization
│   ├── test_metrics.py              # Tests for EEG metrics calculations
│   ├── test_processing.py           # Tests for data processing logic
│
│── utils/                           # Utility functions and helpers
│   ├── __init__.py                  # Module initialization
│   ├── config.py                    # Global settings for file paths and parameters
│   ├── file_io.py                   # Functions for loading and saving data
│   ├── helpers.py                    # Miscellaneous helper functions
│
│── .gitignore                       # Files to exclude from version control
│── README.md                        # Project documentation
│── pyproject.toml                   # Python build system configuration
│── requirements.txt                  # Required Python dependencies
│── setup.py                         # Setup file for package installation
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
## Project Setup
To set up this project correctly, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Create a Conda Environment
It is recommended to use a virtual environment to manage dependencies. Create one using Conda:
```bash
conda create --name eeg-analysis-env python=3.10
conda activate eeg-analysis-env
```

### 3. Install Dependencies
Install the required dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Install the Package in Editable Mode
Since `eeg_analyzer/` and `utils/` contain reusable modules, install them in **editable mode** to allow modifications without reinstalling:
```bash
pip install -e . --use-pep517
```
This ensures that `eeg_analyzer` and `utils` are available for import in all scripts and notebooks.

### 5. Verify Installation
Test that everything is set up correctly by running:
```bash
python -c "import eeg_analyzer; print('EEG Analyzer imported successfully')"
```

If you see the success message, the project is properly configured.

## Usage
You can now run scripts inside `scripts/` or use Jupyter notebooks inside `notebooks/`.

### **Running a Script in `scripts/`**
```bash
python scripts/dir_tree.py
```

---
## Troubleshooting
If you encounter issues with imports, try the following:
1. Ensure the correct Conda environment is activated:
   ```bash
   conda activate eeg-analysis-env
   ```
2. If you installed the package earlier, you may need to reinstall:
   ```bash
   pip install -e . --use-pep517
   ```
3. Run the script from the **root directory** (where `setup.py` is located).

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

