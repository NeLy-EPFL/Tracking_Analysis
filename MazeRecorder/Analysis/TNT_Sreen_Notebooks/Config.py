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

from scipy import stats
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
from statsmodels.stats.multitest import multipletests


from tqdm import tqdm

### Configuration for the analysis

datapath = Utils.get_data_server()

# Import the Split registry
SplitRegistry = pd.read_csv(datapath / "MD/Region_map_250130.csv")

# Downsampling if any

downsampling_factor = None

# Statistics

show_signif = True

# Data binning

data_bins = 10

# Permutation test parameters

permutation_method = "scipy"

permutation_metric = "avg_distance"

n_permutations = 1000

error_logs = True

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

    if brain_region_data.empty: 
        raise ValueError(f"Empty dataset for {brain_region}")
    if control_region_data.empty:
        raise ValueError(f"Empty control dataset")
    
    # Combine the datasets
    combined_data = pd.concat([brain_region_data, control_region_data], ignore_index=True)

    # Downsample the data if downsample_factor is provided
    if downsample_factor:
        combined_data = combined_data.iloc[::downsample_factor, :]

    return combined_data

def preprocess_data(data, bins=10):
    """
    Prepare a simplified dataset by splitting the data into n time bins and computing the average and median distance for each bin. It should be done individually for each fly.

    Args:
        data (pd.DataFrame): The input data containing 'time', 'Brain region', 'fly', and 'distance_ball_0'.
        bins (int, optional): The number of time bins to split the data into. Default is 10.

    Returns:
        pd.DataFrame: A DataFrame containing the average and median distance for each time bin, brain region, and fly.
    """
    # Create a new column 'time_bin' that indicates the time bin each 'time' value belongs to
    data['time_bin'] = pd.cut(data['time'], bins=bins, labels=False)

    # TODO: Implement bins = None to use the original time values
    
    # Group by 'time_bin', 'Brain region', and 'fly' and compute the average and median distance for each group
    avg_distance = data.groupby(['time_bin', 'Brain region', 'fly'])['distance_ball_0'].mean().reset_index()
    avg_distance.columns = ['time_bin', 'Brain region', 'fly', 'avg_distance']

    median_distance = data.groupby(['time_bin', 'Brain region', 'fly'])['distance_ball_0'].median().reset_index()
    median_distance.columns = ['time_bin', 'Brain region', 'fly', 'median_distance']

    # Merge the average and median distance dataframes
    processed_data = avg_distance.merge(median_distance, on=['time_bin', 'Brain region', 'fly'])

    return processed_data

def compute_permutation_test(data, metric, n_permutations=1000, show_progress=False, verbose=False):
    Focal = data[data["Brain region"] != "Control"]
    Control = data[data["Brain region"] == "Control"]
    
    if verbose:
        print(f"Number of Focal data points: {len(Focal)}")
        print(f"Number of Control data points: {len(Control)}")

    # Group by time_bin and fly
    Focal_grouped = Focal.groupby(['time_bin', 'fly'])[metric].mean().unstack()
    Ctrl_grouped = Control.groupby(['time_bin', 'fly'])[metric].mean().unstack()
    
    if verbose:
        print(f"Number of Focal timepoints: {len(Focal_grouped)}")
        print(f"Number of Control timepoints: {len(Ctrl_grouped)}")
        print(f"{Focal_grouped.head()}")
        print(f"{Ctrl_grouped.head()}")

    observed_diffs = Focal_grouped.mean(axis=1) - Ctrl_grouped.mean(axis=1)
    
    p_values = []
    for time_bin in Focal_grouped.index:
        focal_bin = Focal_grouped.loc[time_bin]
        ctrl_bin = Ctrl_grouped.loc[time_bin]
        
        observed_diff = focal_bin.mean() - ctrl_bin.mean()
        combined = np.concatenate([focal_bin, ctrl_bin])
        n_focal = len(focal_bin)
        
        perm_diffs = []
        for _ in range(n_permutations):
            perm = np.random.permutation(combined)
            perm_diff = perm[:n_focal].mean() - perm[n_focal:].mean()
            perm_diffs.append(perm_diff)
        
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        p_values.append(p_value)

    p_values = np.array(p_values)
    
    # Apply multiple testing correction
    _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    
    significant_timepoints = np.where(p_values < 0.05)[0]
    significant_timepoints_corrected = np.where(p_values_corrected < 0.05)[0]

    results = {
        'observed_diff': observed_diffs,
        'p_values': p_values,
        'significant_timepoints': significant_timepoints,
        'p_values_corrected': p_values_corrected,
        'significant_timepoints_corrected': significant_timepoints_corrected,
        'n_significant': len(significant_timepoints),
        'percent_significant': len(significant_timepoints) / len(p_values) * 100,
        'n_significant_corrected': len(significant_timepoints_corrected),
        'percent_significant_corrected': len(significant_timepoints_corrected) / len(p_values) * 100
    }
    
    if verbose:
        print(f"Observed differences: {observed_diffs}")
        print(f"P-values: {p_values}")
        print(f"Significant timepoints: {significant_timepoints}")
        print(f"P-values (corrected): {p_values_corrected}")
        print(f"Significant timepoints (corrected): {significant_timepoints_corrected}")

    return results




def create_and_save_plot(data, nicknames, brain_region, output_path, registries, show_signif=False, show_progress=False, test_nickname=None):
    if test_nickname:
        nicknames = [test_nickname]
    
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
        
        print(f"Processing {nickname} vs {associated_control}")
        
        # Preprocess the data for permutation test
        preprocessed_data = preprocess_data(subset_data, bins=data_bins)
        
        if show_signif:
            permutation = compute_permutation_test(preprocessed_data, permutation_metric, n_permutations=n_permutations, show_progress=show_progress, verbose=error_logs)
            
            # Add the 'Significant' column to the preprocessed data
            preprocessed_data['Significant'] = preprocessed_data['time_bin'].isin(permutation['significant_timepoints_corrected'])
        
        # Plot the raw data
        sns.lineplot(data=subset_data, x='time', y='distance_ball_0', hue='Brain region', ax=axes[i], palette=color_dict)
        
        # Highlight significant timepoints with dotted lines and asterisks
        if show_signif:
            try:
                significant_bins = preprocessed_data[preprocessed_data['Significant']]['time_bin']
                for time_bin in range(data_bins):
                    bin_start = subset_data['time'].min() + time_bin * (subset_data['time'].max() - subset_data['time'].min()) / data_bins
                    bin_end = bin_start + (subset_data['time'].max() - subset_data['time'].min()) / data_bins
                    
                    # Draw faint dotted lines for the bins
                    axes[i].axvline(bin_start, color='gray', linestyle='dotted', alpha=0.5)
                    axes[i].axvline(bin_end, color='gray', linestyle='dotted', alpha=0.5)
                    
                    # Annotate significance levels
                    if time_bin in significant_bins.values:
                        p_value = permutation['p_values_corrected'][time_bin]
                        if p_value < 0.001:
                            significance = '***'
                        elif p_value < 0.01:
                            significance = '**'
                        elif p_value < 0.05:
                            significance = '*'
                        else:
                            significance = ''
                        
                        if significance:
                            # Lower the y position of the stars and increase font size
                            y_position = subset_data['distance_ball_0'].max() * 0.95
                            axes[i].annotate(significance, xy=(bin_start + (bin_end - bin_start) / 2, y_position), 
                                            xycoords='data', ha='center', va='bottom', fontsize=16, color='red')
                
                axes[i].set_title(f'{nickname} vs {associated_control}')
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel('Ball distance from start (px)')
                axes[i].legend().remove()  # Remove the legend
            
            except Exception as e:
                print(f"Error: {e}")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free up memory
    
def create_control_plot(data, control_nicknames, output_path):
    """
    Create and save a plot for all control nicknames together.
    
    Args:
    data (pd.DataFrame): The input data containing 'Brain region', 'time', and 'distance_ball_0'.
    control_nicknames (list): List of control nicknames.
    output_path (str): Path to save the output plot.
    """
    # Subset data for all control nicknames
    control_data = data[data['Nickname'].isin(control_nicknames)]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=control_data, x='time', y='distance_ball_0', hue='Nickname')
    
    plt.title('Control Brain Region')
    plt.xlabel('Time (s)')
    plt.ylabel('Ball distance from start (px)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory

    
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
    
def create_and_save_contact_rate_plot(data, nicknames, brain_region, output_path, registries, color_dict):
    # Define the time window (e.g., 100 seconds)
    time_window = 100

    # Determine the maximum time in the dataset
    max_time = data['start'].max()

    # Create bins for the time windows
    bins = np.arange(0, max_time + time_window, time_window)

    # Create a new column 'time_window' that indicates the time window each 'start' time belongs to
    data['time_window'] = pd.cut(data['start'], bins=bins, right=False, labels=bins[:-1])

    if brain_region == "Control":
        # Special case: Plot all controls together
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Subset data for all control nicknames
        control_data = data[data['Nickname'].isin(nicknames)]
        
        # Group by 'time_window', 'Nickname', and 'fly' and count the number of unique contacts in each time window
        contact_rate = control_data.groupby(['time_window', 'Nickname', 'fly'])['identifier'].nunique().reset_index()
        contact_rate.columns = ['time_window', 'Nickname', 'fly', 'contact_count']
        
        # Plot the contact rate over time with confidence intervals
        sns.lineplot(data=contact_rate, x='time_window', y='contact_count', hue='Nickname', marker='o', ax=ax, ci='sd')
        ax.set_title('Contact Rate Over Time: All Controls', fontsize=12)
        ax.set_xlabel('Time Window (s)')
        ax.set_ylabel('Number of Unique Contacts')
        ax.tick_params(labelsize=10)
        
    else:
        # Default behavior for other brain regions
        pairs_per_row = 3
        n_nicknames = len(nicknames)
        n_rows = math.ceil(n_nicknames / pairs_per_row)
        
        subplot_size = (10, 6)  # Adjusted size for each subplot
        fig_width, fig_height = subplot_size[0] * pairs_per_row, subplot_size[1] * n_rows
        
        fig, axes = plt.subplots(n_rows, pairs_per_row, figsize=(fig_width, fig_height), squeeze=False)
        
        for i, nickname in enumerate(nicknames):
            row = i // pairs_per_row
            col = i % pairs_per_row
            
            nickname_data = data[data['Nickname'] == nickname]
            split_value = nickname_data['Split'].iloc[0]
            associated_control = registries["control_nicknames_dict"][split_value]
            control_data = data[data['Nickname'] == associated_control]
            subset_data = pd.concat([nickname_data, control_data])
            
            # Group by 'time_window', 'Brain region', and 'fly' and count the number of unique contacts in each time window
            contact_rate = subset_data.groupby(['time_window', 'Brain region', 'fly'])['identifier'].nunique().reset_index()
            contact_rate.columns = ['time_window', 'Brain region', 'fly', 'contact_count']
            
            # Plot the contact rate over time with confidence intervals
            sns.lineplot(data=contact_rate, x='time_window', y='contact_count', hue='Brain region', 
                         marker='o', palette=color_dict, ci='sd', ax=axes[row, col])
            axes[row, col].set_title(f'{nickname} vs {associated_control}', fontsize=10)
            axes[row, col].set_xlabel('Time Window (s)')
            axes[row, col].set_ylabel('Number of Unique Contacts')
            axes[row, col].tick_params(labelsize=8)
            axes[row, col].legend(title='Brain region', fontsize=8, title_fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory