from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../../..")

from Utilities.Utils import *
from Utilities.Processing import *


DataPath = Path("/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Datasets")

Dataset = pd.read_feather(DataPath / "231010_DatasetTNT.feather")

GroupOps = Dataset.groupby(
    [
        "time",
        "Genotype",
    ]
)

Confints = GroupOps["yball_relative"].apply(lambda x: draw_bs_ci(x, n_reps=300))


Confints_df = Confints.to_frame()

Confints_df_reset = Confints_df.reset_index()
Confints_df_reset.to_feather(DataPath / "231010_Confints.feather")

notify_me()