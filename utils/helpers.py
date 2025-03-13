import os

def print_directory_tree(root_dir, indent="", file_limit=10, exclude_dirs=[".git"]):
    """
    Recursively scans and prints the hierarchical directory structure, excluding specified directories.

    Args:
        root_dir (str): Root directory to scan.
        indent (str): Indentation for hierarchical display (used recursively).
        file_limit (int): Limits the number of files displayed per directory.
        exclude_dirs (list): Directories to exclude from the printout.
    """
    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        print(indent + "[Access Denied]")
        return

    dirs = [entry for entry in entries if os.path.isdir(os.path.join(root_dir, entry)) and entry not in exclude_dirs]
    files = [entry for entry in entries if os.path.isfile(os.path.join(root_dir, entry))]

    if len(files) > file_limit:
        files = files[:file_limit] + ["... ({} more)".format(len(files) - file_limit)]

    for d in dirs:
        print(f"{indent}📂 {d}/")
        print_directory_tree(os.path.join(root_dir, d), indent + "│   ", file_limit, exclude_dirs)

    for f in files:
        print(f"{indent}📄 {f}")

