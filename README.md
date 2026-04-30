# SANA — Smart Autonomous Natural Agent

Edge-based Eutrophication Detection & Response System. Fog-Computing Swarm Simulation with Agentic AI Controller.

## Features
- **Real-time Telemetry**: Simulated LoRa RX stream from 4 autonomous nodes.
- **AI Controller**: Agentic AI (via Ollama) analyzing water quality indices.
- **Automated Response**: AI-driven intervention (Aerators, Chemical deployment).
- **Luxury UI**: High-fidelity dashboard built with CustomTkinter.

## Prerequisites
- **Python 3.10+**
- **Ollama**: [Download Ollama](https://ollama.com/)
  - Pull the default model: `ollama pull gemma3:1b`

## Quick Start (Windows)

1. **Setup**: Run the setup script to create a virtual environment and install dependencies.
   ```powershell
   .\setup.ps1
   ```

2. **Run**: Launch the dashboard.
   ```powershell
   .\run.ps1
   ```

## Quick Start (Raspberry Pi / Linux)

1. **Setup**: Run the setup script. This will also attempt to install `python3-tk` if missing.
   ```bash
   chmod +x setup.sh run.sh
   ./setup.sh
   ```

2. **Run**: Launch the dashboard.
   ```bash
   ./run.sh
   ```

> [!NOTE]
> On Raspberry Pi, if the GUI feels sluggish, ensure you have allocated enough GPU memory in `raspi-config` or are using a Pi 4/5.

## Repository Structure
- `sana_dashboard.py`: Main Ollama-based dashboard.
- `sana_openrouter.py`: Alternative edition using OpenRouter API.
- `requirements.txt`: Python dependencies.
- `setup.ps1` / `run.ps1`: Automation scripts.
