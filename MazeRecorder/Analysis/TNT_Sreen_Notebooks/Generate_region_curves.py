from pathlib import Path
import json
import pyarrow
import math
import re
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from scipy.stats import gaussian_kde
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import holoviews as hv
from holoviews import opts

hv.extension("bokeh")

from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from utils_behavior import (
    Sleap_utils,
    HoloviewsTemplates,
    Utils,
    Processing,
    Ballpushing_utils,
    Seaborn_Templates,
)

import traceback

import Config

from tqdm import tqdm

import importlib

# Process brain regions

registries = Config.registries
data_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240120_short_contacts_no_cutoff_no_downsample_Data/coordinates_regions")
output_dir = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Plots/250130_Coordinates_Full_Signif")

print(f"Available brain regions: {registries['brain_regions']}")
print(f"Data path contents: {list(data_path.glob('*.feather'))}")


# Initialize miscellaneous data collection
miscellaneous_data = pd.DataFrame()
miscellaneous_nicknames = []

for brain_region in registries["brain_regions"]:
    
    if brain_region != "Control":
        output_path = output_dir / f"Full_euclidean_distance_coordinates_line_{brain_region}.png"
    
        if os.path.exists(output_path):
            print(f"Skipping Brain region {brain_region} as the plot already exists.")
            continue
        
        BallTrajectories = Config.load_datasets_for_brain_region(brain_region, data_path, registries, downsample_factor=None)
        if BallTrajectories.empty:  # Use pandas.DataFrame.empty check
            print(f"Skipping {brain_region} - empty dataset")
            continue
        
        # Get focal-specific data for nickname filtering
        focal_data = BallTrajectories[BallTrajectories['Brain region'] == brain_region]
        nicknames = focal_data['Nickname'].unique()
        
        
        if len(nicknames) < 5 and brain_region != "Control":
            # Collect both nicknames AND their corresponding data
            misc_data = BallTrajectories[BallTrajectories['Nickname'].isin(nicknames)]
            miscellaneous_data = pd.concat([miscellaneous_data, misc_data])
            miscellaneous_nicknames.extend(nicknames)
            continue
        try:
            Config.create_and_save_plot(
                BallTrajectories,  
                nicknames, 
                brain_region, 
                output_path, 
                registries, 
                show_signif=True
            )
        except Exception as e:
            print(f"Error processing {brain_region}: {str(e)}")
            traceback.print_exc()
            
    else:
        output_path = output_dir / "Full_euclidean_distance_coordinates_line_Control.png"
        
        if os.path.exists(output_path):
            
            print(f"Skipping Control Brain region as the plot already exists.")
            continue
        
        BallTrajectories = Config.load_datasets_for_brain_region(brain_region, data_path, registries, downsample_factor=None)


        nicknames = BallTrajectories['Nickname'].unique()
        
        # Plot the Control brain region
        try:
            control_nicknames = registries["control_nicknames"]
            
            Config.create_control_plot(BallTrajectories, control_nicknames, output_path)
            print("Processed Control Brain region")
        except Exception as e:
            print(f"Error processing Control Brain region: {e}")

# Process Miscellaneous brain region
if not miscellaneous_data.empty:
    try:
        output_path = output_dir / "Full_euclidean_distance_coordinates_line_Miscellaneous.png"
        if os.path.exists(output_path):
            print("Skipping Miscellaneous Brain region as the plot already exists.")
        else:
            print(f"Processing Miscellaneous Brain region with {len(miscellaneous_nicknames)} nicknames")
            
            # Ensure we have control data for all miscellaneous nicknames
            combined_data = pd.concat([
                miscellaneous_data,
                Config.load_datasets_for_brain_region("Control", data_path, registries)
            ])
            
            Config.create_and_save_plot(
                combined_data,  # Use the accumulated data + control
                miscellaneous_nicknames, 
                "Miscellaneous", 
                output_path, 
                registries, 
                show_signif=True
            )
            print("Processed Miscellaneous Brain region")
    except Exception as e:
        print(f"Error processing Miscellaneous Brain region: {e}")
        traceback.print_exc()
else:
    print("No miscellaneous nicknames to process")