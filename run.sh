#!/bin/bash

# SANA Run Script for Raspberry Pi / Linux

if [ ! -d "venv" ]; then
    echo -e "\033[0;31m[ERROR] Virtual environment not found! Run ./setup.sh first.\033[0m"
    exit 1
fi

echo -e "\033[0;36m[INFO] Launching SANA Dashboard...\033[0m"
./venv/bin/python3 sana_dashboard.py
