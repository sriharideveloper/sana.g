# SANA Setup Script for Windows (PowerShell)

Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  SANA — Smart Autonomous Natural Agent Setup              ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# 1. Check if Python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python not found! Please install Python 3.10+ and add it to PATH." -ForegroundColor Red
    exit
}

# 2. Create Virtual Environment
if (!(Test-Path "venv")) {
    Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Green
    python -m venv venv
} else {
    Write-Host "[INFO] Virtual environment already exists." -ForegroundColor Yellow
}

# 3. Install Dependencies
Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Green
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  SETUP COMPLETE!                                         ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  To run the dashboard:                                   ║"
Write-Host "║  .\run.ps1                                               ║"
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Green
