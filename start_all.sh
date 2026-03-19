#!/bin/bash

echo "Starting AI Meeting Assistant..."

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 1. Start the Transcriber in the background (using '&')
python transcriber_service.py &
# Capture the Process ID of the transcriber
TRANSCRIBER_PID=$!

echo "Transcriber service running in the background (PID: $TRANSCRIBER_PID)."
echo "Starting the Recorder..."

# 2. Start the Voice Recorder (this blocks until you close the recorder window)
python recorder.py

# 3. Cleanup: Once you close the recorder, automatically stop the background transcriber
echo "Recorder closed. Stopping the background transcriber service..."
kill $TRANSCRIBER_PID
echo "All services completely shut down."
