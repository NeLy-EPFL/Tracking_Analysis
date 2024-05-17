import sys

# Add the Utilities directory to the path

sys.path.insert(0, "../..")

from Utilities.Utils import *
from Utilities.Ballpushing_utils import *

# Get the data path
datapath = get_data_path()

print(f"Data path: {datapath}")

# Generate a list of experiments to consider


Folders = []
for folder in datapath.iterdir():
    minfolder = str(folder).lower()
    if "tnt_fine" in minfolder:
        Folders.append(folder)

print(Folders)

# For each folder, generate an Experiment object
Experiments = []
for folder in Folders:
    Experiments.append(Experiment(folder))

# Get a list of Fly objects in the video for which the genotype contains either "M6" or "M7" or "Z2035"

Flies = []
for experiment in Experiments:
    for fly in experiment.flies:
        if "M6" in fly.Genotype or "M7" in fly.Genotype or "Z2035" in fly.Genotype:
            
            fly.generate_preview(save=True)

# for each fly, generate a preview of the video