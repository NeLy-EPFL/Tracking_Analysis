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



#import lux

import pandas as pd

#lux.config.set_executor_type("Pandas")

import numpy as np
import h5py
import re

# Get the folders to be analyzed

# Get data path and TNT folders

data_path = Utils.get_data_path()

experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/Experiments")

savepath = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/Contact_data")

#tnt_folders = [folder for folder in data_path.iterdir() if folder.is_dir() and 'TNT_Fine' in folder.name]

#print(f" Folders to analyse : {tnt_folders}")

# For each folder, generate the experiment object and then make a dataset out of it, then save it to csv

# for folder in tnt_folders:
#     # Check if a dataset with the same name already exists :
    
#     dataset_name = folder.name + ".csv"
    
#     # If it does, skip the folder
#     # if (savepath/dataset_name).exists():
#     #     print(f"Dataset {dataset_name} already exists. Skipping folder {folder.name}")
#     #     continue

#     if (savepath / f"{folder.name}.pkl").exists():
#         print(f"Experiment {folder.name} already exists. Skipping folder {folder.name}")
#         continue
    
#     experiment = Ballpushing_utils.Experiment(folder)
    
#     try:
#         Ballpushing_utils.save_object(experiment, savepath / f"{folder.name}.pkl")
#     except Exception as e:
#         print(f"Could not save experiment {folder.name}")
#         print(e)
#         continue
    
#     dataset = Ballpushing_utils.Dataset(experiment)
    
#     try:
#         dataset.generate_dataset(metrics="Skeleton_contacts")
#         dataset.data.to_csv(savepath / dataset_name)
#         print(f"Dataset {dataset_name} saved.")
#     except Exception as e:
#         print(f"Could not generate dataset for {folder.name}")
#         print(e)
#         continue

# # Then, concatenate all the datasets into one big dataset

# datasets = [pd.read_csv(file) for file in savepath.iterdir() if file.suffix == ".csv"]

# big_dataset = pd.concat(datasets)

# big_dataset.to_csv(savepath/"241204_Pooled.csv")

Exps = experiment_path.glob("*.pkl")

for experiment_path in Exps:
    print(f"loading {experiment_path}")
    
    dataset_path = savepath / f"{experiment_path.stem}_contact_data.csv"
    
    if dataset_path.exists():
        print(f"Dataset {dataset_path} already exists. Skipping experiment {experiment_path.stem}")
        continue
    
    try:
        experiment = Ballpushing_utils.load_object(experiment_path)
    except Exception as e:
        print(f"Could not load experiment {experiment_path}")
        print(e)
        continue
    
    try:
        dataset = Ballpushing_utils.Dataset(experiment, dataset_type="contact_data")
        dataset.data.to_csv(dataset_path)
        print(f"Dataset {dataset_path} saved.")
    except Exception as e:
        print(f"Could not generate dataset for {experiment_path.stem}")
        print(e)
        continue
    

# Then, concatenate all the datasets into one big dataset

datasets = [pd.read_csv(file) for file in savepath.iterdir() if file.suffix == ".csv"]

big_dataset = pd.concat(datasets)

big_dataset.to_csv(savepath/"241206_Pooled_contact_data.csv")
    