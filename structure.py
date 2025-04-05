import os

def print_directory_tree(startpath, indent=""):
    # Get the list of all files and directories in the given path
    items = os.listdir(startpath)
    items.sort()
    for index, item in enumerate(items):
        path = os.path.join(startpath, item)
        # Choose a prefix symbol for better tree visualization
        connector = "└── " if index == len(items) - 1 else "├── "
        print(indent + connector + item)
        # If the item is a directory, recursively print its contents
        if os.path.isdir(path):
            extension = "    " if index == len(items) - 1 else "│   "
            print_directory_tree(path, indent + extension)

if __name__ == "__main__":
    # Print the structure starting from the current directory
    print_directory_tree(".")
