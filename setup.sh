#!/usr/bin/env bash
#
# Set up the project directory for execution, and download the data if needed.
#
# NOTE: Before you can use this script to download the data, you need to install
#       the Kaggle API. Instructions for doing so can be found here:
#       https://github.com/Kaggle/kaggle-api

# Make sure we are in the root of the project directory; otherwise exit
if [ ! ${PWD##*/} = "salt-identification" ]; then
    echo "This script needs to be run from the data/ directory."
    exit 1
fi

# Make sure that the necessary directories have been created
if [ ! -d "output" ]; then mkdir "output"; fi
if [ ! -d "data" ]; then mkdir "data"; fi
if [ ! -d "data/train" ]; then mkdir "data/train"; fi
if [ ! -d "data/test" ]; then mkdir "data/test"; fi

# Check if the data is missing
missing_data=false
declare -a dirs=("data/train/images" "data/train/masks" "data/test/images")
for dir in "${dirs[@]}"; do
    if [ ! -d ${dir} ]; then mkdir "$dir"; fi
    if [ -z "$(ls -A ${dir})" ]; then missing_data=true; fi
done

# If the data is missing, download it from Kaggle
if [ ${missing_data} = true ]; then
    kaggle competitions download tgs-salt-identification-challenge -p data/
    if [[ (! -f "data/train.zip") || (! -f "data/test.zip") ]]; then exit 1; fi
    unzip data/train.zip -d data/train/
    unzip data/test.zip -d data/test/
    rm data/train.zip
    rm data/test.zip
fi
