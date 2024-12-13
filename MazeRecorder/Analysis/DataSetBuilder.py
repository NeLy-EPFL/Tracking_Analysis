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
experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241209_Exps")

final_event_cutoff_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241213_FinalEventCutoff")
final_event_cutoff_data_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241213_FinalEventCutoffData")

# Check if these folders exist and if not, create them
final_event_cutoff_path.mkdir(parents=True, exist_ok=True)
final_event_cutoff_data_path.mkdir(parents=True, exist_ok=True)

tnt_folders = [folder for folder in data_path.iterdir() if folder.is_dir() and 'TNT_Fine' in folder.name]

print(f" Folders to analyse : {tnt_folders}")

# For each folder, generate the experiment object and then make a dataset out of it, then save it to feather
for folder in tnt_folders:
    experiment_pkl_path = final_event_cutoff_path / f"{folder.name}.pkl"
    dataset_path = final_event_cutoff_data_path / f"{folder.name}_contact_data.feather"
    
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
    
    # Check if the dataset exists
    if dataset_path.exists():
        print(f"Dataset {dataset_path} already exists. Skipping experiment {folder.name}")
        continue
    
    # Generate the dataset from the experiment
    try:
        dataset = Ballpushing_utils.Dataset(experiment, dataset_type="contact_data")
        dataset.data.to_feather(dataset_path)
        print(f"Dataset {dataset_path.name} saved.")
    except Exception as e:
        print(f"Could not generate dataset for {folder.name}")
        print(e)
        continue

# Then, concatenate all the datasets into one big dataset if the pooled dataset doesn't already exist
pooled_dataset_path = final_event_cutoff_data_path / "241209_Pooled_contact_data.feather"
if not pooled_dataset_path.exists():
    try:
        datasets = [pd.read_feather(file) for file in final_event_cutoff_data_path.iterdir() if file.suffix == ".feather"]
        big_dataset = pd.concat(datasets)
        big_dataset.to_feather(pooled_dataset_path)
        print(f"Pooled dataset saved as {pooled_dataset_path}")
    except Exception as e:
        print(f"Could not concatenate datasets: {e}")
else:
    print(f"Pooled dataset {pooled_dataset_path} already exists.")

# Exps = experiment_path.glob("*.pkl")

# for experiment_path in Exps:
#     print(f"loading {experiment_path}")
    
#     dataset_path = savepath / f"{experiment_path.stem}_contact_data.csv"
    
#     if dataset_path.exists():
#         print(f"Dataset {dataset_path} already exists. Skipping experiment {experiment_path.stem}")
#         continue
    
#     try:
#         experiment = Ballpushing_utils.load_object(experiment_path)
#     except Exception as e:
#         print(f"Could not load experiment {experiment_path}")
#         print(e)
#         continue
    
#     try:
#         dataset = Ballpushing_utils.Dataset(experiment, dataset_type="contact_data")
#         dataset.data.to_csv(dataset_path)
#         print(f"Dataset {dataset_path} saved.")
#     except Exception as e:
#         print(f"Could not generate dataset for {experiment_path.stem}")
#         print(e)
#         continue
    

# # Then, concatenate all the datasets into one big dataset

# datasets = [pd.read_csv(file) for file in savepath.iterdir() if file.suffix == ".csv"]

# big_dataset = pd.concat(datasets)

# big_dataset.to_csv(savepath/"241206_Pooled_contact_data.csv")
    