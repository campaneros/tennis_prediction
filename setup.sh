#!/bin/bash

# If venv does NOT exist → create it and install requirements
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv

    echo "[INFO] Activating environment and installing requirements..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[INFO] venv already exists. No installation needed."
fi

# Activate environment every time
source venv/bin/activate
