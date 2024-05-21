#!/bin/bash
script_dir="$(dirname "$0")"

miniconda3_bin="/home/matthias/miniconda3/bin/"

export PATH="$miniconda3_bin:$PATH"

# activate the right conda environment
source activate processing

# execute the check crops command
python "$script_dir/Array2Corridors.py"
