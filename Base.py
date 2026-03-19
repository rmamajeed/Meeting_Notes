pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt (Note: WhisperX requires git to be installed on your system).CPU-only version of PyTorch before installing WhisperX


python transcriber_service.py


# 1. Install the missing venv package (will ask for your password)
sudo apt update && sudo apt install python3.12-venv -y

# 2. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install CPU-only PyTorch FIRST
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install WhisperX and the other requirements
pip install -r requirements.txt



python3 -m venv venv
source venv/bin/activate
./setup_cpu.sh  # (or ./setup_gpu.sh if you want to use the NVIDIA GPU)
