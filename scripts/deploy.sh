#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the scripts directory
cd "$SCRIPT_DIR"

# Check if deploy.py exists and is executable
if [ ! -f "deploy.py" ]; then
    echo "Error: deploy.py not found in scripts directory"
    exit 1
fi

if [ ! -x "deploy.py" ]; then
    echo "Warning: deploy.py is not executable. Making it executable..."
    chmod +x deploy.py
fi

# Forward all arguments to the Python script
./deploy.py "$@" 