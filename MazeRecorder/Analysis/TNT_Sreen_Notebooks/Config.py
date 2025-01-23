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

from tqdm import tqdm


datapath = Utils.get_data_server()

# Import the Split registry
SplitRegistry = pd.read_csv(datapath / "MD/Region_map_250116.csv")

def map_split_registry(df):
    # Add the Split column from the SplitRegistry to the Disp_Data, merging based on the Genotype column, only keeping the Simplified Nickname and Split columns
    df = df.merge(
        SplitRegistry[["Genotype", "Simplified Nickname", "Split"]],
        on="Genotype",
        how="left",
    )

    # Check for NA values in the Split column
    if df["Split"].isna().sum() > 0:
        print(
            f"The Nicknames with NA values in the Split column are: {df[df['Split'].isna()]['Nickname'].unique()}"
        )

    return df


def cleanup_data(df):
    # Remove the non-TNT data from the Disp_Data
    df = df[~df["Genotype"].isin(["M6", "M7", "PR", "CS"])]
    df = df[df["Brain region"] != "None"]
    return df


# Prepare the control data for the analysis
## Manually create a color dictionary for the brain regions
color_dict = {
    "MB": "#1f77b4",  # Blue
    "Vision": "#ff7f0e",  # Orange
    "LH": "#2ca02c",  # Green
    "Neuropeptide": "#d62728",  # Red
    "Olfaction": "#9467bd",  # Purple
    "MB extrinsic neurons": "#8c564b",  # Brown
    "CX": "#e377c2",  # Pink
    "Control": "#7f7f7f",  # Gray
    "None": "#bcbd22",  # Yellow-green
    "fchON": "#17becf",  # Cyan
    "JON": "#ffbb78",  # Light orange
}


def prepare_registries(SplitRegistry):
    brain_regions = SplitRegistry["Simplified region"].unique()
    
    # Define the control region and get unique nicknames
    control_region = "Control"
    control_nicknames = SplitRegistry[SplitRegistry["Simplified region"] == control_region]["Nickname"].unique()
    nicknames = SplitRegistry["Nickname"].unique()
    nicknames = [
        nickname for nickname in nicknames if nickname not in control_nicknames
    ]

    # Combine all nicknames for consistent coloring
    all_nicknames = list(control_nicknames) + nicknames
    control_nicknames_dict = {"y": "Empty-Split", "n": "Empty-Gal4", "m": "TNTxPR"}
    nicknames = SplitRegistry["Nickname"].unique()
    nicknames = [
        nickname
        for nickname in nicknames
        if nickname not in control_nicknames_dict.values()
    ]

    # Make a dictionary with the generated variables
    registries = {
        "brain_regions": brain_regions,
        "control_region": control_region,
        "control_nicknames": control_nicknames,
        "nicknames": nicknames,
        "all_nicknames": all_nicknames,
        "control_nicknames_dict": control_nicknames_dict,
    }

    return registries


registries = prepare_registries(SplitRegistry)

def get_subset_data(df, col="Nickname", value="random"): 
    
    control_nicknames_dict = registries["control_nicknames_dict"]
    
    if value == "random":
        # Pick one random Nickname and get a subset of it
        nicknames = df["Nickname"].unique()
        nickname = np.random.choice(nicknames)
    else:
        nickname = value

    print(f"Nickname selected: {nickname}")

    # Check if 'Split' column exists
    if 'Split' not in df.columns:
        print("Error: 'Split' column not found in dataframe.")
        return pd.DataFrame()

    # Check if the nickname exists in the dataframe
    if nickname not in df["Nickname"].values:
        print(f"Nickname {nickname} not found in dataframe.")
        return pd.DataFrame()

    # Get the associated control
    split_value = SplitRegistry[SplitRegistry["Nickname"] == nickname]["Split"].iloc[0]
    associated_control = control_nicknames_dict.get(split_value)

    if not associated_control:
        print(f"No associated control found for split value {split_value}.")
        return pd.DataFrame()

    print(f"Associated control is: {associated_control}")

    # Get the subset of the data for the random Nickname
    subset_data = df[df["Nickname"] == nickname]

    # Get the subset of the data for the associated control
    control_data = df[df["Nickname"] == associated_control]

    # Check if either subset is empty and handle accordingly
    if subset_data.empty:
        print(f"No data found for nickname {nickname}.")
    if control_data.empty:
        print(f"No data found for associated control {associated_control}.")

    # Combine the nickname data with the relevant control data
    subset_data = pd.concat([subset_data, control_data])

    return subset_data

def load_datasets_for_brain_region(brain_region, data_path, registries, downsample_factor=None):
    # Load the dataset for the brain region
    brain_region_file = data_path / f"{brain_region}.feather"
    if not brain_region_file.exists():
        print(f"Dataset for brain region {brain_region} not found.")
        return pd.DataFrame()

    brain_region_data = pd.read_feather(brain_region_file)

    # Load the dataset for the control region
    control_region = registries["control_region"]
    control_region_file = data_path / f"{control_region}.feather"
    if not control_region_file.exists():
        print(f"Dataset for control region {control_region} not found.")
        return pd.DataFrame()

    control_region_data = pd.read_feather(control_region_file)

    # Combine the datasets
    combined_data = pd.concat([brain_region_data, control_region_data], ignore_index=True)

    # Downsample the data if downsample_factor is provided
    if downsample_factor:
        combined_data = combined_data.iloc[::downsample_factor, :]

    return combined_data


def create_and_save_plot(data, nicknames, brain_region, output_path, registries, show_progress=False):
    n_nicknames = len(nicknames)
    n_cols = 5
    n_rows = math.ceil(n_nicknames / n_cols)
    subplot_size = (6, 6)
    fig_width, fig_height = subplot_size[0] * n_cols, subplot_size[1] * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    
    iterator = enumerate(nicknames)
    if show_progress:
        iterator = tqdm(iterator, total=n_nicknames, desc="Creating subplots")
    
    for i, nickname in iterator:
        nickname_data = data[data['Nickname'] == nickname]
        split_value = nickname_data['Split'].iloc[0]
        associated_control = registries["control_nicknames_dict"][split_value]
        control_data = data[data['Nickname'] == associated_control]
        subset_data = pd.concat([nickname_data, control_data])
        
        sns.lineplot(data=subset_data, x='time', y='distance_ball_0', hue='Brain region', ax=axes[i], palette=color_dict)
        axes[i].set_title(f'{nickname} vs {associated_control}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Median Euclidean Distance')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free up memory
    
# Function to create and save KDE and ECDF plots
def create_and_save_kde_ecdf_plot(data, nicknames, brain_region, output_path, registries):
    if brain_region == "Control":
        # Special case: Plot all controls together
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # One row: KDE and ECDF side by side
        
        # Subset data for all control nicknames
        control_data = data[data['Nickname'].isin(nicknames)]
        
        # KDE Plot
        sns.histplot(data=control_data, x='start', hue="Nickname", ax=axes[0], element="step", kde=True)
        axes[0].set_title('KDE: All Controls', fontsize=12)
        axes[0].set_xlim(0, 3600)
        axes[0].tick_params(labelsize=10)
        
        # ECDF Plot / cumulative distribution
        sns.histplot(data=control_data, x='start', hue="Nickname", ax=axes[1], cumulative=True, element="step", kde=True)
        axes[1].set_title('Cumulative distribution: All Controls', fontsize=12)
        axes[1].set_xlim(0, 3600)
        axes[1].tick_params(labelsize=10)
        
        # Add vertical line between KDE and ECDF
        fig.add_subplot(111, frameon=False)
        plt.vlines(x=0.5, ymin=0, ymax=1, transform=fig.transFigure, colors='black', linewidth=2)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        
    else:
        # Default behavior for other brain regions
        pairs_per_row = 3
        n_nicknames = len(nicknames)
        n_rows = math.ceil(n_nicknames / pairs_per_row)
        n_cols = pairs_per_row * 2  # Two plots (KDE and ECDF) for each pair
        
        subplot_size = (10,6)  # Adjusted size for each subplot
        fig_width, fig_height = subplot_size[0] * pairs_per_row, subplot_size[1] * n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        
        for i, nickname in enumerate(nicknames):
            row = i // pairs_per_row
            col = (i % pairs_per_row) * 2
            
            nickname_data = data[data['Nickname'] == nickname]
            split_value = nickname_data['Split'].iloc[0]
            associated_control = registries["control_nicknames_dict"][split_value]
            control_data = data[data['Nickname'] == associated_control]
            subset_data = pd.concat([nickname_data, control_data])
            
            # KDE Plot
            sns.histplot(data=subset_data, x='start', hue='Brain region', ax=axes[row, col], kde=True, element="step", palette=color_dict)
            axes[row, col].set_title(f'KDE: {nickname}\nvs {associated_control}', fontsize=10)
            axes[row, col].set_xlim(0, 3600)
            axes[row, col].tick_params(labelsize=8)
            
            # ECDF Plot
            sns.histplot(data=subset_data, x='start', hue='Brain region', ax=axes[row, col+1], palette=color_dict, cumulative=True, element="step", kde=True)
            axes[row, col+1].set_title(f'Cumulative: {nickname}\nvs {associated_control}', fontsize=10)
            axes[row, col+1].set_xlim(0, 3600)
            axes[row, col+1].tick_params(labelsize=8)
        
        # # Add vertical and horizontal separators
        # for i in range(1, pairs_per_row):
        #     plt.vlines(x=i/pairs_per_row, ymin=0, ymax=1, transform=fig.transFigure, colors='black', linewidth=2)
        
        # for i in range(1, n_rows):
        #     plt.hlines(y=1-i/n_rows, xmin=0, xmax=1, transform=fig.transFigure, colors='black', linewidth=2)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between subplots
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory