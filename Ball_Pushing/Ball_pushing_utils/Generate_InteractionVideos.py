import sys

# Add the Utilities directory to the path

sys.path.insert(0, "../..")

sys.path.insert(0, "../../Utilities")

import Utils
import Ballpushing_utils

# Get the data path
datapath = Utils.get_data_path()

print(f"Data path: {datapath}")

# Generate a list of experiments to consider


Folders = []
for folder in datapath.iterdir():
    minfolder = str(folder).lower()
    if "tnt_fine" in minfolder and "tracked" in minfolder:
        Folders.append(folder)

print(Folders)

# For each folder, generate an Experiment object
Experiments = []
for folder in Folders:
    Experiments.append(Ballpushing_utils.Experiment(folder))

# # Load the experiments from a file

# # loadpath = (
# #     get_labserver()
# #     / "Experimental_data/MultiMazeRecorder/Datasets/240129_TNT_Fine_Experiments.pkl"
# # )

# #Experiments = load_object(loadpath.as_posix())

# For each fly in each experiment, generate the interaction video and save it in a folder (which might need to be created) depending on its genotype

savepath = Utils.get_labserver() / "Videos" / "240129_TNT_Fine"

# Create the folder if it doesn't exist
if not savepath.exists():
    savepath.mkdir()

for experiment in Experiments:
    for fly in experiment.flies:
        # Check if the folder for the genotype exists
        genotype_path = savepath / fly.Genotype
        if not genotype_path.exists():
            genotype_path.mkdir()
        # Check if the video already exists
        vidname = f"{fly.name}_{fly.Genotype if fly.Genotype else 'undefined'}.mp4"
        video_path = genotype_path / vidname
        if not video_path.exists():
            # If the video doesn't exist, generate it
            fly.generate_interactions_video(outpath=genotype_path)
        
        else:
            print(f"The video {vidname} already exists")
