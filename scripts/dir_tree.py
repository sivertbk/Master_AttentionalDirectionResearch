import sys
# print(sys.path) # We can re-enable this if needed for debugging
from utils.config import ROOT_PATH, DATA_PATH
from utils.helpers import print_directory_tree

# Print ut filstrukturen
print_directory_tree(root_dir=ROOT_PATH, max_depth=3)

print(sys.path)  # Print the current Python path for debugging