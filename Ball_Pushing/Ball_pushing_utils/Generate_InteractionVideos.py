import sys

# Add the Utilities directory to the path

sys.path.insert(0, "../..")

sys.path.insert(0, "../../Utilities")

from Utils import *
from Ballpushing_utils import *

# # Get the data path
# datapath = get_data_path()

# print(f"Data path: {datapath}")

# # Generate a list of experiments to consider


# Folders = []
# for folder in datapath.iterdir():
#     minfolder = str(folder).lower()
#     if "tnt_fine" in minfolder and "tracked" in minfolder:
#         Folders.append(folder)

# print(Folders)

# # For each folder, generate an Experiment object
# Experiments = []
# for folder in Folders:
#     Experiments.append(Experiment(folder))

# Load the experiments from a file

loadpath = (
    get_labserver()
    / "Experimental_data/MultiMazeRecorder/Datasets/240129_TNT_Fine_Experiments.pkl"
)

Experiments = load_object(loadpath.as_posix())

# For each fly in each experiment, generate the interaction video and save it in a folder (which might need to be created) depending on its genotype

savepath = get_labserver() / "Videos" / "240129_TNT_Fine"

# Create the folder if it doesn't exist
if not savepath.exists():
    savepath.mkdir()

for experiment in Experiments:
    for fly in experiment.flies:
        # Check if the folder for the genotype exists
        if not (savepath / fly.Genotype).exists():
            (savepath / fly.Genotype).mkdir()
        fly.generate_interactions_video(outpath=savepath / fly.Genotype)
