# Master Attentional Direction Research

## Project Overview
This project aims to analyze EEG data to study attentional direction in participants. The focus is on processing **Power Spectral Density (PSD)** data derived from EEG signals, using a structured and modular approach in Python.

The analysis workflow includes **preprocessing**, **frequency domain transformation**, and **statistical analysis**, with a focus on **alpha-band activity (8-12 Hz)** as a potential biomarker for attentional shifts.

---

## Project Structure
```
attentional_direction_research_workspace/
📂 data/                                # 📊 All datasets, relevant derivatives, and EEGAnalyzer-objects live here
│   ├── braboszcz2017/                  #   EEG study (meditation vs. thinking)
│   ├── jin2019/                        #   SART + visual-search study
│   ├── touryan2022/                    #   Simulated-driving study
│   └── eeg_analyzer_derivatives/       #   Pickled analyzer states & summary CSVs
│
📂 eeg_analyzer/                        # 🧠 Core Python package (OO analysis engine)
|   ├── eeg_analyzer.py                 #   Top-level object used for analysis of all data
│   ├── dataset.py                      #   Dataset abstraction
│   ├── subject.py                      #   PSD normalisation & filtering logic
│   ├── recording.py.py                 #   Helper metrics (band power, z-scores, …)
│   └── …                               #   subject.py, statistics.py, visualizer.py, …
│
📂 notebooks/                           # 📒 Interactive exploration (Jupyter)
│   ├── braboszcz2017/                  #   Dataset-specific EDA & modelling
│   ├── jin2019/
│   ├── touryan2022/
│   └── shared/                         #   Re-usable analyses (e.g. PSD parameter sweeps)
│
📂 scripts/                             # ⚙️  Command-line automation
│   ├── preprocessing/                  #   AutoReject, ICA, epoching, …
│   ├── analysis/                       #   Stats & plotting & analysis scripts
│   └── dataset-specific/               #   One-off utilities (probe extraction, etc.)
│
📂 reports/                             # 📈 Results for papers / slides
│   ├── plots/                          #   Figures from scripts
│   └── logs/                           #   Analysis & preprocessing logs
│
📂 utils/                               # 🛠  Generic helpers & configuration
│   ├── dataset_configs/                #   Per-dataset metadata objects
│   ├── config.py                       #   Global constants (paths, EEG settings, styles)
│   └── helpers.py                      #   Misc. convenience functions
│
📄 README.md                            # 👉 Start here
📄 PROJECT_PROCESS.md                   #   Project diary / decisions log
📄 pyproject.toml / setup.py            #   Installable package metadata
📄 requirements.txt                     #   Exact Python dependencies

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

## Contact & Contributions
If you're interested in contributing, feel free to fork the repository and submit a pull request.

