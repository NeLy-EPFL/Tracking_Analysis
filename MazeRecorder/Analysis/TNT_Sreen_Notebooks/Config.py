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

    df.head()
    # Check for NA values in the Split column

    df["Split"].isna().sum()
    # Check which Nicknames have NA values in the Split column

    df[df["Split"].isna()]["Nickname"].unique()

    print(
        f" The Nicknames with NA values in the Split column are: {df[df['Split'].isna()]['Nickname'].unique()}"
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


def prepare_registries(df):

    brain_regions = df["Brain region"].unique()
    
    # Define the control region and get unique nicknames
    control_region = "Control"
    control_nicknames = df[df["Brain region"] == control_region]["Nickname"].unique()
    nicknames = df["Nickname"].unique()
    nicknames = [
        nickname for nickname in nicknames if nickname not in control_nicknames
    ]

    # Combine all nicknames for consistent coloring
    all_nicknames = list(control_nicknames) + nicknames
    # Define the control region and get unique nicknames
    control_region = "Control"
    control_nicknames_dict = {"y": "Empty-Split", "n": "Empty-Gal4", "m": "TNTxPR"}
    nicknames = df["Nickname"].unique()
    nicknames = [
        nickname
        for nickname in nicknames
        if nickname not in control_nicknames_dict.values()
    ]

    # Make a dictionnary with the generated variables

    registries = {
        "brain_regions": brain_regions,
        "control_region": control_region,
        "control_nicknames": control_nicknames,
        "nicknames": nicknames,
        "all_nicknames": all_nicknames,
        "control_nicknames_dict": control_nicknames_dict,
    }

    return registries


def get_subset_data(
    df,
    col="Nickname",
    value="random",
):

    registries = prepare_registries(df)

    control_nicknames_dict = registries["control_nicknames_dict"]

    if value == "random":
        # Pick one random Nickname and get a subset of it
        # Get unique Nicknames

        nicknames = df["Nickname"].unique()

        # Pick a random Nickname

        nickname = np.random.choice(nicknames)

    else:
        nickname = value

    print(f"Nickname selected: {nickname}")

    # Get the associated control

    split_value = df[df["Nickname"] == nickname]["Split"].iloc[0]

    associated_control = control_nicknames_dict[split_value]

    print(f"Associated control is : {associated_control}")

    # Get the subset of the data for the random Nickname

    subset_data = df[df["Nickname"] == nickname]

    # Get the subset of the data for the associated control

    control_data = df[df["Nickname"] == associated_control]

    # Combine the nickname data with the relevant control data

    subset_data = pd.concat([subset_data, control_data])

    return subset_data

# Function to create and save plot

def create_and_save_plot(data, nicknames, brain_region, output_path, registries):
    n_nicknames = len(nicknames)
    n_cols = 5
    n_rows = math.ceil(n_nicknames / n_cols)
    subplot_size = (6, 6)
    fig_width, fig_height = subplot_size[0] * n_cols, subplot_size[1] * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    
    for i, nickname in enumerate(nicknames):
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

def create_control_plot(data, control_nicknames, output_path):
    fig, ax = plt.subplots(figsize=(18, 18))
    
    for nickname in control_nicknames:
        nickname_data = data[data['Nickname'] == nickname]
        sns.lineplot(data=nickname_data, x='time', y='distance_ball_0', label=nickname, ax=ax, palette=color_dict)
    
    ax.set_title('Control Nicknames')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Median Euclidean Distance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
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

    
def plot_brain_region_data(
    data,
    output_dir,
    control_nicknames_dict,
    color_dict,
    x_col,
    y_col,
    metric=None,
    plot_type="line",
    n_cols=5,
    figsize=(20, 10),
):
    """
    Generic function to plot data for different Brain regions and Nicknames.

    Parameters:
        data (pd.DataFrame): The input data containing columns 'Brain region', 'Nickname', 'Split', etc.
        output_dir (str): Directory to save the plots.
        control_nicknames_dict (dict): Maps Split values to control nicknames.
        color_dict (dict): Maps Brain regions to colors.
        x_col (str): Column for the x-axis.
        y_col (str): Column for the y-axis.
        metric (str, optional): Additional metric for specific plot types (e.g., scatterplot).
        plot_type (str): Type of plot ('line' or 'scatter').
        n_cols (int): Number of columns in the subplot grid.
        figsize (tuple): Figure size.

    Returns:
        None
    """
    # Group data by Brain region
    brain_regions = data["Brain region"].unique()

    for brain_region in brain_regions:
        # Create directory for the current Brain region if it doesn't exist
        directory = os.path.join(output_dir, brain_region)
        os.makedirs(directory, exist_ok=True)

        # Subset data for the current Brain region
        region_data = data[data["Brain region"] == brain_region]

        # Get unique nicknames for the Brain region
        nicknames = region_data["Nickname"].unique()

        # Calculate rows and columns for subplots
        n_nicknames = len(nicknames)
        n_rows = math.ceil(n_nicknames / n_cols)

        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, nickname in enumerate(nicknames):
            # Subset data for the current nickname
            nickname_data = region_data[region_data["Nickname"] == nickname]

            # Determine associated control group
            split_value = nickname_data["Split"].iloc[0]
            associated_control = control_nicknames_dict[split_value]

            # Subset data for the control group
            control_data = data[data["Nickname"] == associated_control]

            # Combine nickname and control data
            subset_data = pd.concat([nickname_data, control_data])

            # Plot based on specified type
            if plot_type == "line":
                sns.lineplot(
                    data=subset_data,
                    x=x_col,
                    y=y_col,
                    hue="Brain region",
                    ax=axes[i],
                    palette=color_dict,
                )
            elif plot_type == "scatter":
                sns.scatterplot(
                    data=subset_data,
                    x=x_col,
                    y=y_col,
                    hue="Brain region",
                    ax=axes[i],
                    palette=color_dict,
                )

            # Set title and labels
            axes[i].set_title(f"{nickname} vs {associated_control}")
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_col)

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and save figure
        plt.tight_layout()
        plot_path = os.path.join(directory, f"{brain_region}_plots.png")
        plt.savefig(plot_path)

        plt.show()
