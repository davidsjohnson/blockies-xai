import os
import sys
from pathlib import Path
import argparse as ap
import json

import pandas as pd

import utils


concept_attrs = ['spherical', 'ill', 'ill_spherical', 'num_diff', 'bending', 
                 'obj_rotation_roll', 'obj_rotation_pitch', 'obj_rotation_yaw', 
                 'position_x','position_y', 'arm_position', 
                 'obj_color', 'bg_color']

# concepts - 
# num_diff: num_diff_1, num_diff_2, num_diff_3
# bending: bending_low, bending_high
# main_bone_shape: main_bone_shape_low, main_bone_high, main_bone_shape_mutation
# seondary_bone_shape: secondary_bone_shape_low, secondary_bone_high, secondary_bone_shape_mutation
# head_pos: extended_head, tucked_head
# color: color_red, color_neutral, color_blue
# bg_color: bg_color_red, bg_color_neutral, bg_color_blue
# roll: roll_low, roll_high
# pitch: pitch_low, pitch_high
# yaw: yaw_low, yaw_high


def get_json_description():
    # {'name': 'tench.n.01', 'definition': 'freshwater dace-like game fish of Europe and western Asia noted for ability to survive outside water', 'offset': 'n01440764'}

    concept_names = ['num_diff_1', 'num_diff_2', 'num_diff_3', 
                     'bending_low', 'bending_high', 
                     'main_bone_shape_low', 'main_bone_shape_high', 'main_bone_shape_mutation',
                     'secondary_bone_shape_low', 'secondary_bone_shape_high', 'secondary_bone_shape_mutation',
                     'extended_head', 'tucked_head',
                     'color_red', 'color_neutral', 'color_blue',
                     'bg_color_red', 'bg_color_neutral', 'bg_color_blue',
                     'roll_low', 'roll_high',
                     'pitch_low', 'pitch_high',
                     'yaw_low', 'yaw_high']
    concept_descriptions = ["Number of secondary bones: 1", 
                            "Number of secondary bones: 2", 
                            "Number of secondary bones: 3", 
                            "Bending: low", 
                            "Bending: high", 
                            "Main bone shape: low", 
                            "Main bone shape: high", 
                            "Main bone shape: mutation", 
                            "Secondary bone shape: low", 
                            "Secondary bone shape: high", 
                            "Secondary bone shape: mutation", 
                            "Head position: extended", 
                            "Head position: tucked", 
                            "Object color: red", 
                            "Object color: neutral", 
                            "Object color: blue", 
                            "Background color: red", 
                            "Background color: neutral", 
                            "Background color: blue", 
                            "Roll: low", 
                            "Roll: high", 
                            "Pitch: low", 
                            "Pitch: high", 
                            "Yaw: low", 
                            "Yaw: high"]
    offset = ['num_diff_1', 'num_diff_2', 'num_diff_3',
              'bending_low', 'bending_high', 
              'main_bone_shape_low', 'main_bone_shape_high', 'main_bone_shape_mutation',
              'secondary_bone_shape_low', 'secondary_bone_shape_high', 'secondary_bone_shape_mutation',
              'extended_head', 'tucked_head',
              'color_red', 'color_neutral', 'color_blue',
              'bg_color_red', 'bg_color_neutral', 'bg_color_blue',
              'roll_low', 'roll_high',
              'pitch_low', 'pitch_high',
              'yaw_low', 'yaw_high']
    
    df = pd.DataFrame({
        'name': concept_names,
        'definition': concept_descriptions,
        'offset': offset
    })
    
    #to json
    json_description = df.to_dict(orient='index')
    return json_description

def get_num_diff_concepts(df):
    # num_diff: num_diff_1, num_diff_2, num_diff_3
    # num_diff is a categorical variable with values 1, 2, 3

    num_diff = df['num_diff']
    num_diff_onehot = pd.get_dummies(num_diff, prefix='num_diff').astype(int)
    assert (num_diff_onehot.sum(axis=1) == 1).all(), "Num diff one-hot encoding is not mutually exclusive."
    return num_diff_onehot

def get_bending_concepts(df):
    # split at .2

    bending = df['bending']
    bending_high = (bending > 0.2).astype(int)
    bending_low = (bending <= 0.2).astype(int)
    bending_onehot = pd.DataFrame({
        'bending_high': bending_high,
        'bending_low': bending_low
    })
    assert (bending_onehot.sum(axis=1) == 1).all(), "Bending one-hot encoding is not mutually exclusive."
    return bending_onehot

def get_main_bone_shape_concepts(df):
    # less than 0.5 is low, between 0.5 and 1 is high, greater than 1 is mutation

    main_bone_shape = df['spherical']
    main_bone_shape_low = (main_bone_shape < 0.5).astype(int)
    main_bone_shape_high = ((main_bone_shape >= 0.5) & (main_bone_shape <= 1)).astype(int)
    main_bone_shape_mutation = (main_bone_shape > 1).astype(int)
    main_bone_shape_onehot = pd.DataFrame({
        'main_bone_shape_low': main_bone_shape_low,
        'main_bone_shape_high': main_bone_shape_high,
        'main_bone_shape_mutation': main_bone_shape_mutation
    })
    assert (main_bone_shape_onehot.sum(axis=1) == 1).all(), "Main bone shape one-hot encoding is not mutually exclusive."
    return main_bone_shape_onehot

def get_secondary_bone_shape_concepts(df):
    # less than 0.5 is low, between 0.5 and 1 is high, greater than 1 is mutation

    secondary_bone_shape = df['ill_spherical']
    secondary_bone_shape_low = (secondary_bone_shape < 0.5).astype(int)
    secondary_bone_shape_high = ((secondary_bone_shape >= 0.5) & (secondary_bone_shape <= 1)).astype(int)
    secondary_bone_shape_mutation = (secondary_bone_shape > 1).astype(int)
    secondary_bone_shape_onehot = pd.DataFrame({
        'secondary_bone_shape_low': secondary_bone_shape_low,
        'secondary_bone_shape_high': secondary_bone_shape_high,
        'secondary_bone_shape_mutation': secondary_bone_shape_mutation
    })
    assert (secondary_bone_shape_onehot.sum(axis=1) == 1).all(), "Secondary bone shape one-hot encoding is not mutually exclusive."
    return secondary_bone_shape_onehot

def get_head_pos_concepts(df):
    # less than 0.5 is extended, greater than 0.5 is tucked

    head_pos = df['arm_position']
    extended_head = (head_pos <= 0.5).astype(int)
    tucked_head = (head_pos > 0.5).astype(int)
    head_pos_onehot = pd.DataFrame({
        'extended_head': extended_head,
        'tucked_head': tucked_head
    })
    assert (head_pos_onehot.sum(axis=1) == 1).all(), "Head position one-hot encoding is not mutually exclusive."
    return head_pos_onehot

def get_color_concepts(df):
    # color: color_red, color_neutral, color_blue
    # bg_color: bg_color_red, bg_color_neutral, bg_color_blue

    obj_color = df['obj_color']
    red = (obj_color < 0.4).astype(int)
    neutral = ((obj_color >= 0.4) & (obj_color <= 0.6)).astype(int)
    blue = (obj_color > 0.6).astype(int)
    obj_color_onehot = pd.DataFrame({
        'color_red': red,
        'color_neutral': neutral,
        'color_blue': blue
    })
    assert (obj_color_onehot.sum(axis=1) == 1).all(), "Object color one-hot encoding is not mutually exclusive."

    bg_color = df['bg_color']
    red = (bg_color < 0.4).astype(int)
    neutral = ((bg_color >= 0.4) & (bg_color <= 0.6)).astype(int)
    blue = (bg_color > 0.6).astype(int)
    bg_color_onehot = pd.DataFrame({
        'bg_color_red': red,
        'bg_color_neutral': neutral,
        'bg_color_blue': blue
    })
    assert (bg_color_onehot.sum(axis=1) == 1).all(), "Background color one-hot encoding is not mutually exclusive."

    return obj_color_onehot, bg_color_onehot

def get_roll_pitch_yaw_concepts(df):
    # low is absolute value less than 0.5, high is absolute value greater than 0.5

    roll = df['obj_rotation_roll']
    roll_low = (roll.abs() < 0.5).astype(int)
    roll_high = (roll.abs() >= 0.5).astype(int)
    roll_onehot = pd.DataFrame({
        'roll_low': roll_low,
        'roll_high': roll_high
    })
    assert (roll_onehot.sum(axis=1) == 1).all(), "Roll one-hot encoding is not mutually exclusive."

    pitch = df['obj_rotation_pitch']
    pitch_low = (pitch.abs() < 0.5).astype(int)
    pitch_high = (pitch.abs() >= 0.5).astype(int)
    pitch_onehot = pd.DataFrame({
        'pitch_low': pitch_low,
        'pitch_high': pitch_high
    })
    assert (pitch_onehot.sum(axis=1) == 1).all(), "Pitch one-hot encoding is not mutually exclusive."

    yaw = df['obj_rotation_yaw']
    yaw_low = (yaw.abs() < 0.5).astype(int)
    yaw_high = (yaw.abs() >= 0.5).astype(int)
    yaw_onehot = pd.DataFrame({
        'yaw_low': yaw_low,
        'yaw_high': yaw_high
    })
    assert (yaw_onehot.sum(axis=1) == 1).all(), "Yaw one-hot encoding is not mutually exclusive."

    return roll_onehot, pitch_onehot, yaw_onehot


def process_concepts(df):
    nd_onehot = get_num_diff_concepts(df)
    bending_onehot = get_bending_concepts(df)
    main_bone_shape_onehot = get_main_bone_shape_concepts(df)
    secondary_bone_shape_onehot = get_secondary_bone_shape_concepts(df)
    head_pos_onehot = get_head_pos_concepts(df)
    obj_color_onehot, bg_color_onehot = get_color_concepts(df)
    roll_onehot, pitch_onehot, yaw_onehot = get_roll_pitch_yaw_concepts(df)
    onehot_df = pd.concat([nd_onehot, bending_onehot, main_bone_shape_onehot, 
                           secondary_bone_shape_onehot, head_pos_onehot, 
                           obj_color_onehot, bg_color_onehot, 
                           roll_onehot, pitch_onehot, yaw_onehot], axis=1)

    return onehot_df


def main(ds_dir, dataset):
    
    df = utils.load_dataframe(ds_dir, dataset)
    onehot_df = process_concepts(df)

    # Save the one-hot encoded dataframe to a CSV file 
    output_file = ds_dir / f'{dataset}_concepts.csv'
    onehot_df.to_csv(output_file, index=False)
    print(f"One-hot encoded concepts saved to {output_file}")

    # Save the JSON description of concepts
    json_description = get_json_description()
    json_file = ds_dir / f'concepts_description.json'
    with open(json_file, 'w') as f:
        json.dump(json_description, f, indent=4)


if __name__ == "__main__":

    # ds_dir = Path('data/blockies_datasets/sick_ones_bendbias_v3_2class_normal')
    # dataset='train'

    parser = ap.ArgumentParser(description="Extract concepts from dataset")
    parser.add_argument('--ds_dir', type=Path, required=True, help='Directory of the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., train, validation, test)')
    args = parser.parse_args()
    ds_dir = args.ds_dir
    dataset = args.dataset
    if not ds_dir.exists():
        print(f"Dataset directory {ds_dir} does not exist. Please check the path.")
        sys.exit(1)
    if not ds_dir.is_dir(): 
        print(f"Dataset directory {ds_dir} is not a directory. Please check the path.")
        sys.exit(1)

    if dataset not in ['train', 'validation', 'test']:
        print(f"Dataset name {dataset} is not valid. Please use 'train', 'val', or 'test'.")
        sys.exit(1)


    print(f"Extracting concepts from dataset {dataset} in directory {ds_dir}...")
    main(ds_dir, dataset)