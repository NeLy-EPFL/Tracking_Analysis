This directory contains code to interface with .pkl files containing centroid tracking data and perform analyses on the data.

# Project

In this project, flies were recorded in the optobot chambers and their behavior was analyzed using DeepLabCut. There are several groups:
* Control flies : flies that only carry the driver or the UAS
* PD flies : flies taht carry the driver and UAS and develop Parkinson's disease - like symptoms
* PD flies with SynjRQ : Flies that express the 'rescue' factor

# Analysis

The data was then analyzed to extract the classic features measured for Parkinson's disease in flies, such as speed, distance traveled, and the number of times the flies stop.

# Packages

The code uses an older version of the `pandas` package. It works with panda 2.0.3  but not 2.1.1. In case when loading data from the .pkl file the following error is raised:
"TypeError: Argument 'placement' has incorrect type (expected pandas._libs.internals.BlockPlacement, got slice)"

downgrading pandas to 2.0.3 should solve the issue. Also, loading the .pkl file with pickle.load did not work and raised: "ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'"

The issue was solved by using the `pd.read_pickle` function to load the .pkl file.


