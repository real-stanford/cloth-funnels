#!/bin/bash

# Define the directory and the file URL
DIRECTORY="./cloth_funnels/cloth_data/"
FILE_URL="https://clothfunnels.cs.columbia.edu/data/cloth3d_pickle.zip"

# Make the directory if it doesn't already exist
mkdir -p $DIRECTORY

# Download the file using wget
wget -P $DIRECTORY $FILE_URL

# Unzip the file
unzip $DIRECTORY/cloth3d_pickle.zip -d $DIRECTORY
