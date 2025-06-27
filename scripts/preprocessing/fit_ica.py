import os
from tqdm import tqdm

from utils.preprocessing_tools import prepare_ica_epochs, ica_fit
from utils.file_io import load_raw_data, save_ica
from utils.config import DATASETS, set_plot_style, EEG_SETTINGS
from utils.helpers import iterate_dataset_items

set_plot_style()


def fit_ica_models(datasets=None, verbose=True, show_plots=False):
    """
    Fit ICA models for all subjects in the specified datasets.
    
    Parameters
    ----------
    datasets : dict, optional
        Dictionary of datasets to process. If None, uses DATASETS from config.
    verbose : bool, optional
        Whether to print verbose output. Default is True.
    show_plots : bool, optional
        Whether to show plots during processing. Default is False.
    """
    print("[INFO] Starting ICA model fitting...")
    
    if datasets is None:
        datasets = DATASETS.copy()
        # Remove datasets that are not ready for ICA fitting
        datasets.pop('braboszcz2017', None)
        datasets.pop('jin2019', None)
        datasets.pop('touryan2022', None)
    
    total_processed = 0
    total_skipped = 0
    
    for dataset, subject, label, item, kwargs in iterate_dataset_items(datasets):
        print(f"[INFO] Processing ICA fit for Subject: {subject}, {label}-{item}")
        
        # Load and prepare raw data
        raw = load_raw_data(
            dataset, 
            subject, 
            **kwargs, 
            verbose=False
        )

        if raw is None:
            print(f"[WARN] No data found for subject {subject} with {label} {item}.")
            total_skipped += 1
            continue
        
        # Prepare epochs for ICA
        epochs = prepare_ica_epochs(
            raw, 
            dataset, 
            EEG_SETTINGS, 
            subject, 
            **kwargs, 
            verbose=verbose,
            min_threshold=300e-6,
            reject_scale_factor=1.5
        )

        # Fit ICA
        ica = ica_fit(
            epochs,
            EEG_SETTINGS,
            verbose=verbose,
        )

        # Save ICA
        save_ica(
            ica,
            dataset,
            subject,
            **kwargs,
            verbose=verbose
        )

        # Save epochs
        save_path = os.path.join(dataset.path_epochs, 'ica_epochs')
        os.makedirs(save_path, exist_ok=True)
        epochs.save(os.path.join(save_path, f"sub-{subject}_{label}-{item}_ica-epo.fif"), overwrite=True)
        
        total_processed += 1
        print(f"[INFO] Completed ICA fit for Subject: {subject}, {label}-{item}")
    
    print(f"[INFO] ICA model fitting completed. Processed: {total_processed}, Skipped: {total_skipped}")
    return {"processed": total_processed, "skipped": total_skipped}


if __name__ == "__main__":
    # Run the function when script is executed directly
    fit_ica_models(verbose=True, show_plots=False)

    