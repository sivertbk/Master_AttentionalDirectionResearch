import os

def print_directory_tree(root_dir, indent="", file_limit=10):
    """
    Scans and prints a hierarchical file structure from a given root directory.
    
    Args:
        root_dir (str): The root directory to scan.
        indent (str): Indentation for hierarchical display (used recursively).
        file_limit (int): Limit on the number of files displayed per directory (if many files).
    """
    try:
        entries = sorted(os.listdir(root_dir))  # Sorting for consistent output
    except PermissionError:
        print(indent + "[Access Denied]")
        return
    
    dirs = [entry for entry in entries if os.path.isdir(os.path.join(root_dir, entry))]
    files = [entry for entry in entries if os.path.isfile(os.path.join(root_dir, entry))]
    
    # Limit file display if there are too many
    if len(files) > file_limit:
        files = files[:file_limit] + ["... ({} more)".format(len(files) - file_limit)]

    for d in dirs:
        if d == ".git":
            continue
        print(f"{indent}📂 {d}/")
        print_directory_tree(os.path.join(root_dir, d), indent + "│   ", file_limit)

    for f in files:
        print(f"{indent}📄 {f}")