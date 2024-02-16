import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.append("/home/durrieu/Tracking_Analysis/Utilities")
from Utilities.Utils import *
from Utilities.Processing import *

from scipy.signal import savgol_filter
import numpy as np

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

        self.duration = self.data['frame'].max()

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
            # Check if the format is "nd" with n being an int
            if "d" in parts[2]:
                age = int(parts[2].replace("d", ""))
            else:
                # In Irene experiments, the only time the age is not in the format "nd" is for the first experiments where the age was 2.
                age = 2
        elif len(parts) == 2:
            # Get the grand grand grand parent directory name for genotype
            genotype = self.directory.parent.parent.parent.name
            sex = "female" if "f" in parts[0] else "male"
            age = int(parts[1].replace("d", ""))
        elif len(parts) == 4:
            # Combine the first and second parts for genotype
            genotype = parts[0] + "_" + parts[1]
            sex = "female" if "f" in parts[2] else "male"
            age = int(parts[3].replace("d", ""))

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

        # Check for duplicated pos_x and pos_y columns and drop the second one
        data = data.loc[:, ~data.columns.duplicated()]
        # TODO: compute savgol filter with VLR params

        # TODO: Check if it's always the right column that is selected

        # Add a name column

        data["fly"] = self.directory.parent.parent.name

        # Add a time column in seconds

        data["time"] = data["frame"] / fps

        data["velocity"] = self.compute_velocity(data)

        # Implement the metadata
        data["genotype"] = self.metadata["genotype"]

        data["sex"] = self.metadata["sex"]

        data["age"] = self.metadata["age"]

        return data

    def compute_velocity(self,data, x_positions=None, y_positions=None, fps = 80, start_frame=None, stop_frame=None, window=25, polyorder=2):
        """
        Compute the velocity between two frames given x and y positions.

        Parameters:
        x_positions (np.array): The x positions of the object.
        y_positions (np.array): The y positions of the object.
        start_frame (int): The frame to start the computation.
        stop_frame (int): The frame to stop the computation.
        fps (int): The frames per second of the video.
        window (int): The window length for the Savitzky-Golay filter. Default is 25.
        polyorder (int): The order of the polynomial for the Savitzky-Golay filter. Default is 2.

        Returns:
        velocity (np.array): The computed velocity between the start and stop frames.
        """
        # If x_positions and y_positions are not provided, use the data from the fly object
        if x_positions is None:
            x_positions = data["pos_x"]

        if y_positions is None:
            y_positions = data["pos_y"]

        # If start_frame or stop_frame is not provided, use the first and last frames respectively
        if start_frame is None:
            start_frame = 0
        if stop_frame is None:
            stop_frame = len(x_positions)-1

        # Ensure start and stop frames are within the length of the positions
        #if start_frame < 0 or start_frame >= len(x_positions) or stop_frame < 0 or stop_frame >= len(x_positions):
        #    raise ValueError("Start and stop frames must be within the length of the positions.")

        # Extract the positions between the start and stop frames
        # x_positions = x_positions[start_frame:stop_frame]
        # y_positions = y_positions[start_frame:stop_frame]

        # Compute the derivatives of the positions using the Savitzky-Golay filter
        dx = savgol_filter(x_positions, window, polyorder, deriv=1, delta=1/fps)
        dy = -savgol_filter(y_positions, window, polyorder, deriv=1, delta=1/fps)

        # Compute the velocity
        velocity = np.sqrt(dx**2 + dy**2) * Optobot_pixelsize

        # pad the velocity to have the same length as the positions
        # if len(velocity) < len(x_positions):
        #     velocity = np.pad(
        #         velocity,
        #         (1, len(x_positions) - len(velocity) - 1),  # Add an additional NaN at the start
        #         mode="constant",
        #         constant_values=float("nan"),
        #     )

        #TODO: Clean this up
        return velocity
