#!/bin/bash
cd "$(dirname "$0")"

miniconda3_bin="/home/matthias/miniconda3/bin/"

export PATH="$miniconda3_bin:$PATH"

# activate the right conda environment
source activate processing


# execute the check crops command
python /home/matthias/Tracking_Analysis/Ball_Pushing/MazeRecorder/Processing/CheckTracks.py