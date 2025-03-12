import os

def print_directory_tree(root_dir, indent="", file_limit=10):
    """
    Skanner og skriver ut en hierarkisk filstruktur fra en gitt rotmappe.
    
    Args:
        root_dir (str): Rotmappen som skal skannes.
        indent (str): Innrykk for hierarkisk visning (brukes rekursivt).
        file_limit (int): Begrensning på antall filer som vises per mappe (hvis mange filer).
    """
    try:
        entries = sorted(os.listdir(root_dir))  # Sortering for konsistent utskrift
    except PermissionError:
        print(indent + "[Access Denied]")
        return
    
    dirs = [entry for entry in entries if os.path.isdir(os.path.join(root_dir, entry))]
    files = [entry for entry in entries if os.path.isfile(os.path.join(root_dir, entry))]
    
    # Begrens filvisning hvis det er for mange
    if len(files) > file_limit:
        files = files[:file_limit] + ["... ({} more)".format(len(files) - file_limit)]

    for d in dirs:
        print(f"{indent}📂 {d}/")
        print_directory_tree(os.path.join(root_dir, d), indent + "│   ", file_limit)

    for f in files:
        print(f"{indent}📄 {f}")