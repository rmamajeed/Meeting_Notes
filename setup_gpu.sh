#!/bin/bash
# setup_gpu.sh
# Installs a full environment optimized for NVIDIA GPU transcription (CUDA 12.1).

echo "Installing GPU-enabled PyTorch (Heavy, ~4GB+)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing WhisperX and other tools..."
pip install -r requirements.txt

echo "GPU Installation Complete. You can now run the service."
