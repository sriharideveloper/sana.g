#!/bin/bash

# SANA Setup Script for Raspberry Pi / Linux
set -e

echo -e "\033[0;36m╔══════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[0;36m║  SANA — Smart Autonomous Natural Agent Setup (Linux/Pi)   ║\033[0m"
echo -e "\033[0;36m╚══════════════════════════════════════════════════════════╝\033[0m"

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "\033[0;31m[ERROR] Python3 not found! Please install it using: sudo apt install python3\033[0m"
    exit 1
fi

# 2. Check for python3-tk (Required for CustomTkinter/Tkinter)
if ! python3 -c "import tkinter" &> /dev/null; then
    echo -e "\033[0;33m[WARN] tkinter not found. This is required for the GUI.\033[0m"
    echo -e "[INFO] Attempting to install python3-tk... (may require password)"
    sudo apt-get update && sudo apt-get install -y python3-tk
fi

# 3. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo -e "\033[0;32m[INFO] Creating virtual environment...\033[0m"
    python3 -m venv venv
else
    echo -e "\033[0;33m[INFO] Virtual environment already exists.\033[0m"
fi

# 4. Install Dependencies
echo -e "\033[0;32m[INFO] Installing dependencies from requirements.txt...\033[0m"
./venv/bin/python3 -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo ""
echo -e "\033[0;32m╔══════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[0;32m║  SETUP COMPLETE!                                         ║\033[0m"
echo -e "\033[0;32m╠══════════════════════════════════════════════════════════╣\033[0m"
echo -e "║  To run the dashboard:                                   ║"
echo -e "║  ./run.sh                                                ║"
echo -e "\033[0;32m╚══════════════════════════════════════════════════════════╝\033[0m"
