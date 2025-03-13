import os
from utils.config import *
from utils.helpers import print_directory_tree

# Angi rotmappen til prosjektet ditt
root_path = os.path.abspath(os.path.join(os.getcwd()))  # Sikrer riktig path

# Print ut filstrukturen
print_directory_tree(ROOT_PATH)


