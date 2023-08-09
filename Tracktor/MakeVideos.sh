#!/bin/bash
cd "$(dirname "$0")"

# activate the right conda environment
source activate trackinganalysis


# execute the check crops command
python /home/matthias/Tracking_Analysis/Tracktor/Images2Vids.py