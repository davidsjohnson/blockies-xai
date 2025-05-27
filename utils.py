import os
import tarfile
import requests
from pathlib import Path

import pandas as pd


def _download_url(url, file_path):
  response = requests.get(url, stream=True)
  response.raise_for_status()  # Raise an error for bad status codes
  with open(file_path, "wb") as file:
      for chunk in response.iter_content(chunk_size=8192):
          file.write(chunk)

def download_file(url, file_name, cache_dir="data", extract=True, force_download=False, archive_folder=None):
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    file_path = Path(cache_dir) / file_name

    # Download the file
    if not os.path.exists(file_path) or force_download:
        print(f"Downloading file from {url} to {file_path}...")
        _download_url(url, file_path)
        print(f"File downloaded to: {file_path}")
    else:
        print(f"File already exists at: {file_path}")

    if extract:
      with tarfile.open(file_path, "r:gz") as tar:
          tar.extractall(path=cache_dir)
      print(f"File extracted to: {cache_dir}")
      file_path = Path(cache_dir) / archive_folder if archive_folder is not None else Path(cache_dir)
    elif not extract and archive_folder is not None:
       file_path = Path(cache_dir) / archive_folder
       print(f"File already extracted to: {file_path}")

    return Path(file_path)

def load_dataframe(data_dir, dataset):
  data_dir = data_dir / dataset
  df = pd.read_json(data_dir / 'parameters.jsonl', lines=True)
  df['filename'] = df['id'] + '.png'
  df['ill'] = df['ill'].astype(int).astype(str)

  return df