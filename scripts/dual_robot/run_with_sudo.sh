#!/bin/bash
# Run dual VR teleoperator with proper conda environment

# Get the conda environment Python path
CONDA_ENV="lefranx"
PYTHON_PATH="/home/amax/miniconda3/envs/$CONDA_ENV/bin/python3"
SCRIPT_PATH="/home/amax/workspace/lerobot/scripts/dual_robot/dual_vr_teleoperator.py"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Set up environment variables for the conda environment
export PATH="/home/amax/miniconda3/envs/$CONDA_ENV/bin:$PATH"
export CONDA_DEFAULT_ENV="$CONDA_ENV"
export CONDA_PREFIX="/home/amax/miniconda3/envs/$CONDA_ENV"
export PYTHONPATH="/home/amax/workspace/lerobot/src:$PYTHONPATH"

echo "Running with Python: $PYTHON_PATH"
echo "Python version: $($PYTHON_PATH --version)"
echo ""

# Run the script
exec "$PYTHON_PATH" "$SCRIPT_PATH"
