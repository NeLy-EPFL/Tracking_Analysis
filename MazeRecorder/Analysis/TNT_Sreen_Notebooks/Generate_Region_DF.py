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

import importlib


# Import the Split registry
datapath = Utils.get_data_server()
SplitRegistry = pd.read_csv(datapath / "MD/Region_map_250116.csv")

# Get all brain regions
brain_regions = SplitRegistry["Simplified region"].unique()
print(f"Brain regions: {brain_regions}")

# Initialize empty DataFrames for each brain region
region_dfs = {region: pd.DataFrame() for region in brain_regions}

# Define input and output directories
input_dir = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240120_short_contacts_no_cutoff_no_downsample_Data/coordinates")
output_dir = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240120_short_contacts_no_cutoff_no_downsample_Data/coordinates_regions")
output_dir.mkdir(parents=True, exist_ok=True)

# Get the list of feather files in the input directory
feather_files = list(input_dir.glob("*.feather"))

# Process each feather file
for file in feather_files:
    data = pd.read_feather(file)
    
    for region in brain_regions:
        region_data = data[data["Brain region"] == region]
        if not region_data.empty:
            region_dfs[region] = pd.concat([region_dfs[region], region_data], ignore_index=True)
            
# Save the region DataFrames to the output directory
for region, df in region_dfs.items():
    df.to_feather(output_dir / f"{region}.feather")