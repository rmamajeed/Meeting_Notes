#!/bin/bash
# setup_cpu.sh
# Installs a lightweight environment optimized for CPU-only transcription.

echo "Installing CPU-only PyTorch (Lightweight)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing WhisperX and other tools..."
pip install -r requirements.txt

echo "CPU Installation Complete. You can now run the service."
