import time
import os
import json
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch

# Configuration from .env
RECORDINGS_DIR = os.environ.get("WATCH_FOLDER", "recordings")
MODEL_NAME = os.environ.get("MODEL_SIZE", "base")

# Hardware / Device Auto-Detection
ENV_DEVICE = os.environ.get("DEVICE_OVERRIDE", "").strip().lower()

if ENV_DEVICE in ["cuda", "cpu"]:
    DEVICE = ENV_DEVICE
    logging.info(f"Hardware device forced to: {DEVICE} via .env")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    logging.info("Auto-detected NVIDIA GPU (CUDA). Running in accelerated mode.")
else:
    DEVICE = "cpu"
    logging.info("No applicable GPU found. Defaulting to CPU mode.")

# Compute type and batch size based on hardware
if DEVICE == "cuda":
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
else:
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))


# Try importing whisperx. If it fails, log an error suggesting how to install.
try:
    import whisperx
except ImportError:
    logging.error("whisperx is not installed. Please install it using: pip install git+https://github.com/m-bain/whisperx.git")
    whisperx = None


class AudioTranscriptionHandler(FileSystemEventHandler):
    def __init__(self, model):
        self.model = model
        self.processing = set()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".mp3"):
            logging.info(f"New audio file detected: {event.src_path}")
            self.process_file_when_ready(event.src_path)

    def on_modified(self, event):
         if not event.is_directory and event.src_path.endswith(".mp3"):
             # Sometimes 'created' fires before the file actually has data.
             # Modification events can trigger processing too.
             if event.src_path not in self.processing:
                 self.process_file_when_ready(event.src_path)

    def process_file_when_ready(self, filepath):
        """Waits for the file to finish writing before processing."""
        if filepath in self.processing:
            return
            
        self.processing.add(filepath)
        
        # Wait until the file size stops changing (meaning the recording is complete)
        logging.info(f"Waiting for {filepath} to finish writing...")
        previous_size = -1
        while True:
            if not os.path.exists(filepath):
                 self.processing.remove(filepath)
                 return
                 
            current_size = os.path.getsize(filepath)
            if current_size == previous_size and current_size > 0:
                break
            previous_size = current_size
            time.sleep(2) # Wait 2 seconds between checks
            
        logging.info(f"File {filepath} is ready for transcription.")
        self.transcribe_audio(filepath)
        self.processing.remove(filepath)

    def transcribe_audio(self, filepath):
        if whisperx is None:
            logging.error("Cannot transcribe because whisperx is missing.")
            return
            
        logging.info(f"Starting transcription for {filepath} on {DEVICE}...")
        try:
             # Load audio
             audio = whisperx.load_audio(filepath)
             
             # Transcribe
             result = self.model.transcribe(audio, batch_size=BATCH_SIZE)
             logging.info(f"Transcription complete. Starting alignment...")
             
             # Align whisper output (Requires a specific model, we'll use english by default)
             model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
             result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
             
             logging.info("Alignment complete. Starting Speaker Diarization...")
             
             hf_token = os.environ.get("HF_TOKEN")
             if hf_token:
                 logging.info("Starting Speaker Diarization...")
                 from whisperx.diarize import DiarizationPipeline
                 diarize_model = DiarizationPipeline(token=hf_token, device=DEVICE)
                 diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=5)
                 result = whisperx.assign_word_speakers(diarize_segments, result)
             else:
                 logging.warning("No HF_TOKEN environment variable found. Skipping Speaker Diarization.")
             
             # Save output
             output_file = os.path.splitext(filepath)[0] + ".txt"
             with open(output_file, "w", encoding="utf-8") as f:
                  for segment in result["segments"]:
                       start = segment.get("start", 0)
                       end = segment.get("end", 0)
                       text = segment.get("text", "")
                       speaker = segment.get("speaker", "UNKNOWN_SPEAKER") if hf_token else "SPEAKER"
                       f.write(f"[{start:.2f} - {end:.2f}] {speaker}: {text.strip()}\n")
                       
             logging.info(f"Saved transcription to {output_file}")
             
        except Exception as e:
             logging.error(f"Error processing {filepath}: {e}")

def main():
    if whisperx is None:
         logging.warning("WhisperX is not installed. The watchdog will still monitor, but won't transcribe.")
         model = None
    else:
         logging.info(f"Loading WhisperX '{MODEL_NAME}' model on {DEVICE} (Compute: {COMPUTE_TYPE})...")
         model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
         logging.info("Model loaded successfully.")

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    
    event_handler = AudioTranscriptionHandler(model)
    observer = Observer()
    observer.schedule(event_handler, path=RECORDINGS_DIR, recursive=False)
    observer.start()
    logging.info(f"Started monitoring '{RECORDINGS_DIR}' directory...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopping Watchdog service...")
    observer.join()

if __name__ == "__main__":
    main()
