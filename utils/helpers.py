import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import ICA
import gc

def print_directory_tree(
    root_dir=None,
    indent="",
    file_limit=10,
    dir_limit=10,
    exclude_dirs=None,
    max_depth=None,
    current_depth=0,
    skip_folders=None,
    output_lines=None,
    save_to_file=False,
    filename="directory_structure.txt"
):
    """
    Prints or saves a directory tree from the root_dir with various options.

    Args:
        root_dir (str): Starting directory (defaults to current working directory).
        indent (str): Indentation string for pretty print.
        file_limit (int): Max number of files per folder to show.
        dir_limit (int): Max number of folders per folder to show.
        exclude_dirs (list): List of dir names to exclude completely.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current level (used internally for recursion).
        skip_folders (list): List of folder names to skip diving into.
        output_lines (list): Internal list of lines being built.
        save_to_file (bool): If True, writes to file instead of printing.
        filename (str): Filename to save the output (if save_to_file=True).
    """
    if root_dir is None:
        root_dir = os.getcwd()

    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__"]

    if skip_folders is None:
        skip_folders = ["data"]

    if output_lines is None:
        output_lines = []

    if max_depth is not None and current_depth > max_depth:
        output_lines.append(f"{indent}... (max depth reached)")
        return output_lines

    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        output_lines.append(f"{indent}🔒 [Access Denied]")
        return output_lines

    dirs = [e for e in entries if os.path.isdir(os.path.join(root_dir, e)) and e not in exclude_dirs]
    files = [e for e in entries if os.path.isfile(os.path.join(root_dir, e))]

    display_dirs = dirs[:dir_limit]
    remaining_dirs = len(dirs) - len(display_dirs)

    display_files = files[:file_limit]
    remaining_files = len(files) - len(display_files)

    for d in display_dirs:
        output_lines.append(f"{indent}📂 {d}/")
        full_path = os.path.join(root_dir, d)
        if d in skip_folders:
            output_lines.append(f"{indent}│   ... (skipped)")
        else:
            print_directory_tree(
                full_path,
                indent + "│   ",
                file_limit,
                dir_limit,
                exclude_dirs,
                max_depth,
                current_depth + 1,
                skip_folders,
                output_lines,
                save_to_file,
                filename
            )

    if remaining_dirs > 0:
        output_lines.append(f"{indent}... ({remaining_dirs} more directories)")

    for f in display_files:
        output_lines.append(f"{indent}📄 {f}")

    if remaining_files > 0:
        output_lines.append(f"{indent}... ({remaining_files} more files)")

    # Final output
    if current_depth == 0:
        if save_to_file:
            with open(filename, "w") as f:
                f.write("\n".join(output_lines))
            print(f"✅ Directory structure saved to: {filename}")
        else:
            for line in output_lines:
                print(line)

    return output_lines

def perform_ica_cleaning(epochs, subject, n_components=20, method='infomax', random_state=97):
    """
    Perform ICA on the provided epochs and allow manual selection of components for exclusion.

    Parameters:
        epochs (mne.Epochs): The epochs object to clean.
        n_components (int): Number of ICA components to compute (default: 20).
        method (str): ICA method to use ('infomax' by default).
        random_state (int): Random seed for ICA initialization (default: 42).

    Returns:
        mne.Epochs: Cleaned epochs after ICA.
        ICA: The ICA object used for the analysis.
    """
    print("Fitting ICA...")
    # Initialize ICA
    ica = ICA(n_components=n_components, method=method, random_state=random_state)

    # Fit ICA to the epochs
    ica.fit(epochs)
    print("ICA fitting complete.")

    # Plot ICA components
    print("Displaying ICA components. Use the interactive GUI to select components for exclusion.")
    ica.plot_components(title=f"ICA decomposition on subject {subject}") 

    # Plot sources to identify artifacts
    print("Displaying ICA sources. Use this to examine the time-series for artifact-related components.")
    ica.plot_sources(epochs, block = True)  # Opens an interactive source plot

    # Apply ICA to remove the selected components
    print("Applying ICA to remove selected components...")
    cleaned_epochs = ica.apply(epochs.copy())
    print("ICA cleaning complete.")

    return cleaned_epochs, ica

def format_numbers(numbers, width=3):
    """
    Format a number or a list of numbers to strings with leading zeros.

    Parameters:
        numbers (int, float, str, list): The number(s) to format.
        width (int): The width of the formatted string (default: 3).

    Returns:
        str or list: Formatted number(s) as string(s).
    """
    if isinstance(numbers, (int, float)):
        return f"{numbers:0{width}d}"
    elif isinstance(numbers, str):
        return f"{int(numbers):0{width}d}"
    elif isinstance(numbers, list):
        return [f"{int(num):0{width}d}" for num in numbers]
    else:
        raise TypeError("Input must be an int, float, str, or list of these types.")

# Function to calculate frequency resolution based on sampling rate and epoch length
def calculate_freq_resolution(epoch_length_sec):
    """
    Calculate frequency resolution (Rayleigh frequency).

    Parameters:
        epoch_length_sec (float): Length of epoch in seconds.

    Returns:
        float: Frequency resolution in Hz.
    """
    return 1.0 / epoch_length_sec

if __name__ == "__main__":
    print("Helper functions loaded successfully.")