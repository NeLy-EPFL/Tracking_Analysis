import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.append("/home/durrieu/Tracking_Analysis/Utilities")
from Utilities.Utils import *
from Utilities.Processing import *

def find_experiments(directory):
    """
    Function to find all experiments in a directory. Experiments are folders containing .mp4 files and a "resultsDLC" folders.

    Args:
        directory (pathlib path): The directory to search for experiments.
    """
    
    experiments = [x for x in directory.iterdir() if x.is_dir()]
    experiments = [x for x in experiments if len(list(x.glob("*.mp4"))) > 0]
    experiments = [x for x in experiments if len(list(x.glob("resultsDLC"))) > 0]

    return experiments

class Fly:
    """
    A class to represent one fly in the experiment. It contains methods to load relevant information about the fly and experiment, along with methods to load DeepLabCut predictions, and to process and analyze the data.
    """
    
    def __init__ (self,
                  directory):
        """ 
            Initializes the Fly object with the directory of the fly's data.

        Args:
            directory (pathlib path): The directory of the fly's data.
        """
        
        