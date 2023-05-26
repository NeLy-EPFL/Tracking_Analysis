import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import more_itertools as mit
from pathlib import Path
import os
from scipy import signal
import holoviews as hv
from holoviews import opts

import re
from tqdm import tqdm
import subprocess

# Path definitions

inputfolder = Path("/home/matthias/Videos/Test2/")

# Load the first frame

frame = cv2.imread(inputfolder.joinpath("image0.jpg").as_posix())

# make it grayscale
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# equalize the histogram to make thresholding easier
frame = cv2.equalizeHist(frame)

# rotate the image 90 degrees
frame = np.rot90(frame)

