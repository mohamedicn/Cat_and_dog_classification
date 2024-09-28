import os

# Folder path
folder_path = r'F:\books\FOUR YEAR\Data Mining\New folder\project\input'

# Count the number of files (photos) in the folder
file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

print(f'Total number of photos in the folder: {file_count}')
