import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.append("/home/durrieu/Tracking_Analysis/Utilities")
from Utilities.Utils import *
from Utilities.Processing import *

# In the optobot, 32 mm  = 832 px

Optobot_pixelsize = 32 / 832  # mm/px

# For this experiments, fps is always the same

fps = 80


def find_experiments(directory):
    """
    Function to find all experiments in a directory. Experiments are folders containing .mp4 files and a .pkl file with the same name as the video.

    Args:
        directory (pathlib path): The directory to search for experiments.
    """

    experiments = []

    for x in directory.iterdir():
        if x.is_dir():
            if len(list(x.glob("*.mp4"))) > 0 and len(list(x.glob("*.pkl"))) > 0:
                experiments.append(x)
            experiments.extend(find_experiments(x))

    return experiments


class Fly:
    """
    A class to represent one fly in the experiment. It contains methods to load relevant information about the fly and experiment, along with methods to load centroid tracking data, with analysis tools.
    """

    def __init__(self, directory):
        """
            Initializes the Fly object with the directory of the fly's data.

        Args:
            directory (pathlib path): The directory of the fly's data.
        """

        self.directory = directory

        self.metadata = self.extract_metadata()

        self.data = self.load_data()

    def extract_metadata(self):
        # Get the grand grand parent directory name
        grand_grand_parent = self.directory.parent.parent.name

        # Split the directory name into parts
        parts = grand_grand_parent.split("_")

        # Check the number of parts
        if len(parts) == 3:
            # Extract the metadata
            genotype = parts[0]
            sex = "female" if "f" in parts[1] else "male"
            age = int(parts[2].replace("d", ""))
        elif len(parts) == 2:
            # Get the grand grand grand parent directory name for genotype
            genotype = self.directory.parent.parent.parent.name
            sex = "female" if "f" in parts[0] else "male"
            age = int(parts[1].replace("d", ""))

        # Return the metadata as a dictionary
        return {"genotype": genotype, "sex": sex, "age": age}

    def load_data(self):
        """
        Loads the data from the .pkl file into a pandas dataframe.
        """
        # Find the .pkl file name

        pkl_file = list(self.directory.glob("*.pkl"))[0]

        data = pd.read_pickle(self.directory / pkl_file)
        # drop multi_index columns
        data.columns = data.columns.droplevel(0)

        data.reset_index(inplace=True)

        # Add a time column in seconds

        data["time"] = data["frame"] / fps

        # Implement the metadata
        data["genotype"] = self.metadata["genotype"]

        data["sex"] = self.metadata["sex"]

        data["age"] = self.metadata["age"]

        return data
