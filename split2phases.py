import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import shutil



CLASSES = ['Healthy', 'OCDegen']

XAICOLS_MAP = {
    'concept': ['Concept1_Salience', 'Concept2_Salience', 'Concept3_Salience', 'Concept4_Salience', 'Concept5_Salience', 'contribution_plot'],
    # 'concept': ['Example1', 'Example2', 'Example3', 'Example4'],
    'example': ['Example1', 'Example2', 'Example3', 'Example4'],
    'cfs': ['Example1', 'Example2', 'Example3'],
    'feature': ['Shap'],
    'none': []
}

def shuffle_dataset(df, n_correct, random_state=9):
  
  assert n_correct % 2 == 0, "n_correct must be even"

  # get n correct instances for beginning of df
  correct_df = df[df['TRUE_DIAG'] == df['SUGGESTED_DIAG']]
  correct_healthy_samples = correct_df[correct_df['TRUE_DIAG'] == CLASSES[0]].sample(n_correct // 2, random_state=random_state)
  correct_ocd_samples = correct_df[correct_df['TRUE_DIAG'] == CLASSES[1]].sample(n_correct // 2, random_state=random_state)
  correct_samples_df = pd.concat([correct_healthy_samples, correct_ocd_samples])
  correct_samples_df = correct_samples_df.sample(frac=1, random_state=random_state)

  # shuffle remaining instances
  remaining_df = df.drop(correct_samples_df.index).reset_index(drop=True)
  remaining_df = remaining_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

  # add correct instances to beginning of df
  final_df = pd.concat([correct_samples_df, remaining_df]).reset_index(drop=True)

  return final_df

def split_dataset(df, random_state=9):
   
   if 'PHASE' in df.columns:
        print("Splitting dataset based on PHASE column.")
        # Split based on PHASE column if it exists
        df1 = df[df['PHASE'] == 1]
        df2 = df[df['PHASE'] == 2]
   else:
        print("Splitting dataset using stratified sampling.")
        # otherwise, perform stratified split
        # Split the df into two equal sets with the same distribution of TRUE_DIAG, SUGGESTED_DIAG combinations
        df1 = df.groupby(['TRUE_DIAG', 'SUGGESTED_DIAG']).sample(frac=0.5, random_state=random_state)
        df2 = df.drop(df1.index)

   return df1, df2

def unzip_images(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted images to {extract_to}")

def zip_images(filelist, img_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for img_name in filelist:
            img_path = img_dir / img_name
            if img_path.exists():
                zipf.write(img_path, arcname=img_name)

    print(f"Saved {len(filelist)} images to {output_zip}")

def _print_nicely(title, df, n=10):
    sep = "-" * 80
    print()
    print(sep)
    print(f"{title}  (shape={df.shape})")
    print(sep)
    print("First rows:")
    print(df.head(n).to_string(index=False))
    print(sep)
    print()


def main(input_folder, output_folder, xai_type, n_correct=4, random_state=9):

    input_csv = input_folder / ("input_example.csv" if xai_type != 'feature' else "input_saliency.csv")
    input_zip = input_folder / "study_images.zip" if xai_type != 'concept' else input_folder / "study_images_example.zip"

    output_folder.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Split the DataFrame into two phases based on 'PHASE' column
    # order by image so splitting preserves that order
    df = df.sort_values(by='X_RAY_IMAGE').reset_index(drop=True)
    phase1_df, phase2_df = split_dataset(df, random_state=random_state)
    phase1_df = shuffle_dataset(phase1_df, n_correct=n_correct, random_state=random_state)
    phase2_df = shuffle_dataset(phase2_df, n_correct=n_correct, random_state=random_state)

    assert phase1_df.iloc[:n_correct]['TRUE_DIAG'].equals(phase1_df.iloc[:n_correct]['SUGGESTED_DIAG']), f"First {n_correct} samples in phase 1 are not correct"
    assert phase2_df.iloc[:n_correct]['TRUE_DIAG'].equals(phase2_df.iloc[:n_correct]['SUGGESTED_DIAG']), f"First {n_correct} samples in phase 2 are not correct"

    # display output
    # Improve console display for pandas DataFrames
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.colheader_justify", "left")
    pd.set_option("display.max_colwidth", 50)


    _print_nicely("Phase 1", phase1_df.groupby(['TRUE_DIAG', 'SUGGESTED_DIAG']).size().reset_index(name='count'), n=10)
    _print_nicely("Phase 2", phase2_df.groupby(['TRUE_DIAG', 'SUGGESTED_DIAG']).size().reset_index(name='count'), n=10)

    # Save the split DataFrames to new CSV files
    phase1_df.to_csv(output_folder / "input_examples_phase1.csv", index=False)
    phase2_df.to_csv(output_folder / "input_examples_phase2.csv", index=False)

    # Unzip images to a temporary directory
    temp_img_dir = output_folder / "temp_images"
    temp_img_dir.mkdir(parents=True, exist_ok=True)
    unzip_images(input_zip, temp_img_dir)

    # Zip images for each phase
    main_filecol = ['X_RAY_IMAGE']
    phase1_files = pd.unique(phase1_df[main_filecol].values.ravel())
    phase2_files = pd.unique(phase2_df[main_filecol].values.ravel())

    xai_filecols = XAICOLS_MAP.get(xai_type, [])
    phase1_xai_files = pd.unique(phase1_df[xai_filecols].values.ravel())
    phase2_xai_files = pd.unique(phase2_df[xai_filecols].values.ravel())

    phase1_all_files = np.concatenate([phase1_files, phase1_xai_files], axis=0)
    phase2_all_files = np.concatenate([phase2_files, phase2_xai_files], axis=0)

    # Zip images for each phase
    zip_images(phase1_all_files, temp_img_dir, output_folder / "images_phase1.zip")
    zip_images(phase2_all_files, temp_img_dir, output_folder / "images_phase2.zip")

    # Clean up temporary image directory

    if temp_img_dir.exists():
        try:
            shutil.rmtree(temp_img_dir)
            print(f"Removed temporary directory {temp_img_dir}")
        except Exception as e:
            print(f"Failed to remove temp images directory {temp_img_dir}: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split dataset into two phases and package images.")
    parser.add_argument("--input_folder", type=Path, required=True, help="Path to the input folder")
    parser.add_argument("--output_folder", type=Path, required=True, help="Output folder path")
    parser.add_argument("--xai_type", type=str, choices=['concept', 'example', 'feature', 'none', 'cfs'], default='example', help="Type of XAI explanations to include")
    parser.add_argument("--n_correct", type=int, default=4, help="Number of correct samples to include in each phase")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, n_correct=args.n_correct, xai_type=args.xai_type)