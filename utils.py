import os
from typing import Union, List, Optional
import tarfile
import requests
from pathlib import Path

from PIL import Image

import pandas as pd


def _download_url(url, file_path):
  response = requests.get(url, stream=True)
  response.raise_for_status()  # Raise an error for bad status codes
  with open(file_path, "wb") as file:
      for chunk in response.iter_content(chunk_size=8192):
          file.write(chunk)

def download_file(url: str, 
                  file_name: str, 
                  cache_dir: str="data", 
                  extract: bool=True,
                  force_download: bool=False, 
                  archive_folder:Optional[str]=None) -> Path:
    """ Download a file from a URL and save it to a specified directory.

    Parameters:
      url (str): The URL of the file to download.
      file_name (str): The name to save the file as.
      cache_dir (str): The directory where the file will be saved.
      extract (bool): Whether to extract the file if it is a tar.gz archive.
      force_download (bool): Whether to force re-download the file if it already exists.
      archive_folder (str, optional): The folder to extract the contents to if the file is an archive.

    Returns:
      Path: The path to the downloaded (and possibly extracted) file.
    """

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

def load_dataframe(data_dir: Union[str, Path], 
                   dataset: str) -> pd.DataFrame:
  """
  Load the blocky dataframe from the specified dataset directory.  
  
  Parameters:
    data_dir (str or Path): The directory where the dataset is stored.  
    dataset (str): The name of the dataset to load.
  
  Returns:
    pd.DataFrame: A DataFrame containing the dataset parameters.
  """

  data_dir = data_dir / dataset
  df = pd.read_json(data_dir / 'parameters.jsonl', lines=True)
  df['filename'] = df['id'] + '.png'

  df['ill'] = (df['obj_name'] == 'ocd').astype(int)
  
  return df


def load_dataframe_old(data_dir: Union[str, Path], 
                       dataset: str) -> pd.DataFrame:
  """
  Load the blocky dataframe from the specified dataset directory.  used for 
  datasets created with the previous version of Blockies that isn't based on
  object name and already has the 'ill' column.
  
  Parameters:
    data_dir (str or Path): The directory where the dataset is stored.  
    dataset (str): The name of the dataset to load.
  
  Returns:
    pd.DataFrame: A DataFrame containing the dataset parameters.
  """

  data_dir = data_dir / dataset
  df = pd.read_json(data_dir / 'parameters.jsonl', lines=True)
  df['filename'] = df['id'] + '.png'
  
  return df

def load_imgs(paths: List[Union[str, Path]]) -> List[Image.Image]:
  """ Load images from a list of file paths.
  Parameters: 
    paths (List[Union[str, Path]]): List of file paths to the images.
  
  Returns:  
    List[PIL.Image.Image]: List of loaded images.
  """

  imgs = []
  for path in paths:
      img = Image.open(path).convert('RGB')
      imgs.append(img)
  return imgs


def get_progressive_blockies(bid: int, 
                             exp_df: pd.DataFrame, 
                             feature_name: str='bending') -> pd.DataFrame:
    """
    Get the progressive blockies for a given ID from the xai_df and exp_df.
    
    Parameters:
      id (int): The ID of the blocky.
      exp_df (pd.DataFrame): The DataFrame containing experimental data.
      feature_name (str): The name of the feature used for the experiment(default is 'bending').
    
    Returns:
      pd.DataFrame: A DataFrame containing the progressive blockies for the given ID.
    """
    exp_bids = [f'{bid}_{feature_name}_{i}' for i in range(1, 11)]
    xai_exp_subset = exp_df[exp_df['id'].isin(exp_bids)]
    return xai_exp_subset