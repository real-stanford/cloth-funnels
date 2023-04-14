#!/bin/bash

# Check if the assets folder exists, if not, create it
if [ ! -d "assets" ]; then
  mkdir assets
fi

# Check if the assets/tasks folder exists, if not, create it
if [ ! -d "assets/tasks" ]; then
  mkdir assets/tasks
fi

# Download files
base_url="https://clothfunnels.cs.columbia.edu/data/tasks/"
files=(
  "longsleeve-single.hdf5"
  "multi-longsleeve-eval.hdf5"
  "multi-longsleeve-train.hdf5"
  "longsleeve-single.hdf5.lock"
  "multi-longsleeve-eval.hdf5.lock"
  "multi-longsleeve-train.hdf5.lock"
)

for file in "${files[@]}"; do
  wget -P assets/tasks/ "$base_url$file"
done
