import os
import zipfile
import shutil

def extract_tif_files(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            
            # If it's a zip file, extract TIF files from it
            if file.lower().endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename.lower().endswith('.tif'):
                            zip_ref.extract(zip_info, destination_folder)
                            extracted_file = os.path.join(destination_folder, zip_info.filename)
                            new_name = os.path.join(destination_folder, os.path.basename(extracted_file))
                            os.rename(extracted_file, new_name)
            
            # If it's a TIF file, copy it to the destination folder
            elif file.lower().endswith('.tif'):
                shutil.copy2(file_path, destination_folder)

# Usage
source_folder = '/Users/remi/Documents/IP/patent_data/I20240102'
destination_folder = '/Users/remi/repos/reverse_image_search/bulk_images'

extract_tif_files(source_folder, destination_folder)
print(f"All .TIF files have been extracted to {destination_folder}")