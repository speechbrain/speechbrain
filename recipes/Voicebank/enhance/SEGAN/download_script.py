import os
from voicebank_prepare import download_vctk

# Define destination folder for the dataset
destination_folder = os.path.abspath("/data/datasets")
print(destination_folder)

# Ensure the folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Call the function
download_vctk(destination=destination_folder)