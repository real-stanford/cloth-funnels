#!/bin/bash

# Download longsleeve-single.hdf5
base_url="https://clothfunnels.cs.columbia.edu/data/tasks/"
file="longsleeve-single.hdf5"

# Check if the assets folder exists, if not, create it
if [ ! -d "assets" ]; then
  mkdir assets
fi

# Check if the assets/tasks folder exists, if not, create it
if [ ! -d "assets/tasks" ]; then
  mkdir assets/tasks
fi

wget -P assets/tasks/ "$base_url$file"