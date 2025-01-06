from utils_behavior import Sleap_utils
from utils_behavior import HoloviewsTemplates
from utils_behavior import HoloviewsPlots
from utils_behavior import Utils
from utils_behavior import Processing
from utils_behavior import Ballpushing_utils

import importlib
from pathlib import Path
import json
from matplotlib import pyplot as plt
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import seaborn as sns
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui
import numpy as np
import h5py
import re

# Get the folders to be analyzed

# Get data path and TNT folders
data_path = Utils.get_data_path()
#experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241209_Exps")

final_event_cutoff_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/250106_FinalEventCutoff_norm")
final_event_cutoff_data_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/250106_FinalEventCutoffData_norm")

# Check if these folders exist and if not, create them
final_event_cutoff_path.mkdir(parents=True, exist_ok=True)
final_event_cutoff_data_path.mkdir(parents=True, exist_ok=True)

tnt_folders = [folder for folder in data_path.iterdir() if folder.is_dir() and 'TNT_Fine' in folder.name]

print(f" Folders to analyse : {tnt_folders}")

# Define the list of metrics to generate datasets for
metrics_list = ["coordinates", 
                "contact_data", 
                "summary", 
                # "F1_coordinates", 
                # "F1_summary", 
                # "F1_checkpoints", 
                "Skeleton_contacts"]

# Create directories for each metric
for metric in metrics_list:
    metric_path = final_event_cutoff_data_path / metric
    metric_path.mkdir(parents=True, exist_ok=True)

# For each folder, generate the experiment object and then make datasets out of it, then save them to feather
for folder in tnt_folders:
    experiment_pkl_path = final_event_cutoff_path / f"{folder.name}.pkl"
    
    # Check if datasets need to be generated
    datasets_needed = False
    for metric in metrics_list:
        dataset_path = final_event_cutoff_data_path / metric / f"{folder.name}_{metric}.feather"
        if not dataset_path.exists():
            datasets_needed = True
            break
    
    if not datasets_needed:
        print(f"All datasets for {folder.name} already exist. Skipping experiment.")
        continue
    
    # Check if the experiment has already been generated as a .pkl file
    if experiment_pkl_path.exists():
        print(f"Experiment {folder.name} already exists. Loading experiment.")
        try:
            experiment = Ballpushing_utils.load_object(experiment_pkl_path)
        except Exception as e:
            print(f"Could not load experiment {folder.name}")
            print(e)
            continue
    else:
        print(f"Generating experiment for {folder.name}")
        try:
            experiment = Ballpushing_utils.Experiment(folder, success_cutoff=True, success_cutoff_method="final_event")
            Ballpushing_utils.save_object(experiment, experiment_pkl_path)
        except Exception as e:
            print(f"Could not save experiment {folder.name}")
            print(e)
            continue
    
    # Generate and save datasets for each metric
    for metric in metrics_list:
        dataset_path = final_event_cutoff_data_path / metric / f"{folder.name}_{metric}.feather"
        
        # Check if the dataset exists
        if dataset_path.exists():
            print(f"Dataset {dataset_path} already exists. Skipping experiment {folder.name} for metric {metric}")
            continue
        
        # Generate the dataset from the experiment
        try:
            dataset = Ballpushing_utils.Dataset(experiment, dataset_type=metric)
            if not dataset.data.empty:
                dataset.data.to_feather(dataset_path)
                print(f"Dataset {dataset_path.name} saved.")
            else:
                print(f"No data for {folder.name} with metric {metric}")
        except Exception as e:
            print(f"Could not generate dataset for {folder.name} with metric {metric}")
            print(e)
            continue

# Then, concatenate all the datasets into one big dataset if the pooled dataset doesn't already exist
for metric in metrics_list:
    pooled_dataset_path = final_event_cutoff_data_path / metric / f"250106_Pooled_{metric}.feather"
    if not pooled_dataset_path.exists():
        try:
            datasets = [pd.read_feather(file) for file in (final_event_cutoff_data_path / metric).iterdir() if file.suffix == ".feather"]
            if datasets:
                big_dataset = pd.concat(datasets)
                big_dataset.to_feather(pooled_dataset_path)
                print(f"Pooled dataset saved as {pooled_dataset_path}")
            else:
                print(f"No datasets found for metric {metric}")
        except Exception as e:
            print(f"Could not concatenate datasets for metric {metric}: {e}")
    else:
        print(f"Pooled dataset {pooled_dataset_path} already exists.")