#!/bin/bash
cd "$(dirname "$0")"

miniconda3_bin="/home/matthias/miniconda3/bin/"

export PATH="$miniconda3_bin:$PATH"

# activate the right conda environment
source activate trackinganalysis


# execute the check crops command
python /home/matthias/Tracking_Analysis/Tracktor/CheckTracks.py