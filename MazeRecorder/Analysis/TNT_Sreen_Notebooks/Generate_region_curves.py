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

import Config

from tqdm import tqdm

import importlib

# Process brain regions

registries = Config.registries
data_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240120_short_contacts_no_cutoff_no_downsample_Data/coordinates_regions")
miscellaneous_nicknames = []
# Add a progress bar to the brain regions processing loop
for brain_region in tqdm(registries["brain_regions"], desc="Processing brain regions"):
    output_path = f"/mnt/upramdya_data/MD/MultiMazeRecorder/Plots/250122_Coordinates_full/Full_euclidean_distance_coordinates_line_{brain_region}.png"
    
    if os.path.exists(output_path):
        print(f"Skipping Brain region {brain_region} as the plot already exists.")
        continue
    
    print(f"Processing Brain region {brain_region}")
    print("loading datasets")
    region_data = Config.load_datasets_for_brain_region(brain_region, data_path, registries, downsample_factor=None)
    nicknames = region_data['Nickname'].unique()
    
    if len(nicknames) < 5 and brain_region != "Control":
        miscellaneous_nicknames.extend(nicknames)
        continue
    
    try:
        print("Creating and saving plot")
        Config.create_and_save_plot(region_data, nicknames, brain_region, output_path, registries, show_progress=True)
        print(f"Processed Brain region {brain_region}")
    except Exception as e:
        print(f"Error processing Brain region {brain_region}: {e}")

# Process Miscellaneous brain region
if miscellaneous_nicknames:
    try:
        output_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Plots/250120_coordinates/Full_euclidean_distance_coordinates_line_Miscellaneous.png"
        Config.create_and_save_plot(region_data, miscellaneous_nicknames, "Miscellaneous", output_path, registries, show_progress=True)
        print("Processed Miscellaneous Brain region")
    except Exception as e:
        print(f"Error processing Miscellaneous Brain region: {e}")

# Plot the Control brain region
try:
    control_nicknames = registries["control_nicknames"]
    output_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Plots/250120_coordinates/Full_euclidean_distance_coordinates_line_Control.png"
    Config.create_control_plot(region_data, control_nicknames, output_path)
    print("Processed Control Brain region")
except Exception as e:
    print(f"Error processing Control Brain region: {e}")