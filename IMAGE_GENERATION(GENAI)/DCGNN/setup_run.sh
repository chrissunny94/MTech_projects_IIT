#!/bin/bash

# Stage 1: Check for Nvidia GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "Stage 1: Checking Nvidia GPU information"
    nvidia-smi
fi

# Stage 2: Uninstall TensorFlow (if necessary)
echo "Stage 2: Uninstalling TensorFlow"
pip uninstall tensorflow -y

# Stage 3: Print requirements
echo "Stage 3: Printing requirements.txt"
cat requirements.txt

# Stage 4: Install dependencies
echo "Stage 4: Installing dependencies"
pip install -r requirements.txt

# Stage 5: Run the Python script
echo "Stage 5: Running Python script"
python3 code.py