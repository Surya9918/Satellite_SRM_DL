# Installation and Setup Guide

## System Prerequisites
- Linux (Ubuntu 20.04/22.04 recommended) or Windows 10/11
- Python 3.11+
- NVIDIA GPU (T4, P100, RTX 3080/4090) with CUDA 11.8+ or CPU fallback

## Step-by-Step Installation
```bash
# Clone the repository
git clone https://github.com/ntro-srm/satellite-srm.git
cd satellite-srm

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Run automated system setup
python -m satellite_srm setup
```

## Credentials Setup
Copy `.env.example` to `.env` and configure your Copernicus credentials:
```bash
cp .env.example .env
```
