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
    if "balltype" in minfolder and "tracked" in minfolder:
        Folders.append(folder)

print(Folders)

# For each folder, generate an Experiment object
Experiments = []
for folder in Folders:
    Experiments.append(Ballpushing_utils.Experiment(folder))

#Load the experiments from a file

# loadpath = (
#     Utils.get_labserver()
#     / "Experimental_data/MultiMazeRecorder/Datasets/240306_TNT_Fine_Experiments.pkl"
# )

#Experiments = Ballpushing_utils.load_object(loadpath.as_posix())

# For each fly in each experiment, generate the interaction video and save it in a folder (which might need to be created) depending on its genotype

savepath = Utils.get_labserver() / "Videos" / "BallTypes_Annotated"

# Create the folder if it doesn't exist
if not savepath.exists():
    savepath.mkdir()

for experiment in Experiments:
    for fly in experiment.flies:
        # Check if the folder for the balltype exists
        ballpath = savepath / fly.arena_metadata["BallType"]
        if not ballpath.exists():
            ballpath.mkdir()
        # Check if the video already exists
        vidname = f"{fly.name}_{fly.arena_metadata['BallType'] if fly.arena_metadata['BallType'] else 'undefined'}.mp4"
        video_path = ballpath / vidname
        if not video_path.exists():
            # If the video doesn't exist, generate it
            fly.generate_interactions_video(outpath=ballpath, tracks=True)
        
        else:
            print(f"The video {vidname} already exists")
