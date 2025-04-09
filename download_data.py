import zipfile
import urllib.request
import os
from urllib.parse import urlparse

# List of ZIP file URLs
zip_urls = [
    'https://www.openslr.org/resources/61/es_ar_female.zip',
    'https://www.openslr.org/resources/61/es_ar_male.zip'
]

# Local filenames and destination folders
zip_names = ['female_audio.zip', 'male_audio.zip']
dest_folders = ['./data/female_audio', './data/male_audio']

# Download and extract each ZIP
for url, zip_name, dest in zip(zip_urls, zip_names, dest_folders):
    print(f"Downloading {zip_name} from {url}...")
    urllib.request.urlretrieve(url, zip_name)

    # Create destination folder if it doesn't exist
    os.makedirs(dest, exist_ok=True)

    print(f"Extracting {zip_name} into ./{dest}...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(dest)

    print(f"Deleting {zip_name}...")
    os.remove(zip_name)

# List of TSV URLs
tsv_urls = [
    'https://www.openslr.org/resources/61/line_index_female.tsv',
    'https://www.openslr.org/resources/61/line_index_male.tsv'
]

# Folder to save the files
save_folder = './data'
os.makedirs(save_folder, exist_ok=True)

# Download files using the filename from the URL
for url in tsv_urls:
    filename = os.path.basename(urlparse(url).path)
    local_path = os.path.join(save_folder, filename)

    print(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, local_path)

print("All .tsv files downloaded.")

print("All done.")