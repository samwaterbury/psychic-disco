#!/usr/bin/env bash

# Sets up the project directory for execution, and downloads the data if needed.
# This script will put the project directory in the DEFAULT configuration; see
# the global variable `DEFAULT_PATHS` near the start of `utilities.py`.
#
# NOTE: Before you can use this script to download the data, you need to install
#       the Kaggle API. Instructions for doing so can be found here:
#       https://github.com/Kaggle/kaggle-api

# Make sure we are in the root of the project directory; otherwise exit
if [ ! ${PWD##*/} = "salt-identification" ]; then
    echo "This script needs to be run from the project root directory."
    exit 1
fi

# Make sure that the necessary directories have been created
if [ ! -d "output" ]; then mkdir "output"; fi
if [ ! -d "data" ]; then mkdir "data"; fi
if [ ! -d "data/train" ]; then mkdir "data/train"; fi
if [ ! -d "data/train/images" ]; then mkdir "data/train/images"; fi
if [ ! -d "data/train/masks" ]; then mkdir "data/train/masks"; fi
if [ ! -d "data/test" ]; then mkdir "data/test"; fi
if [ ! -d "data/test/images" ]; then mkdir "data/test/images"; fi

# Check if the data is missing
missing_data=false
if [ -z "$(ls -A data/train/images)" ]; then missing_data=true; fi
if [ -z "$(ls -A data/train/masks)" ]; then missing_data=true; fi
if [ -z "$(ls -A data/test/images)" ]; then missing_data=true; fi
if [ ! -f "data/train.csv" ]; then missing_data=true; fi
if [ ! -f "data/depths.csv" ]; then missing_data=true; fi

# If the data is missing, download it from Kaggle
if [ ${missing_data} = true ]; then
    kaggle competitions download tgs-salt-identification-challenge -p data/
    if [ ! -f "data/train.zip" ]; then exit 1; fi
    unzip -q data/train.zip -d data/train/
    unzip -q data/test.zip -d data/test/
    rm data/train.zip
    rm data/test.zip
fi
