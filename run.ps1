# SANA Run Script for Windows (PowerShell)

if (!(Test-Path "venv")) {
    Write-Host "[ERROR] Virtual environment not found! Run .\setup.ps1 first." -ForegroundColor Red
    exit
}

Write-Host "[INFO] Launching SANA Dashboard..." -ForegroundColor Cyan
.\venv\Scripts\python.exe sana_dashboard.py
