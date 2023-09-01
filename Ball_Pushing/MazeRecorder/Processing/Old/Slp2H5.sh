#!/bin/bash

# activate the right conda environment
source activate sleap

# set input and output folders
input_folder="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Datasets/FirstExp/"
output_folder="/mnt/labserver/DURRIEU_Matthias/Experimental_data/MultiMazeRecorder/Sleap/Datasets/FirstExp/"

# for each .slp file in the input folder
for file in "$input_folder"/*.slp; do
    # extract the filename without extension
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    
    # execute the command to extract and save .h5 files
    sleap-convert "$file" --format analysis
done