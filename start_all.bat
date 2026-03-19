@echo off
echo Starting AI Meeting Assistant...

:: Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: 1. Start the Transcriber in a separate, minimized background window
start "Audio Transcriber Background" /MIN python transcriber_service.py

echo Transcriber service running in the background.
echo Starting the Recorder...

:: 2. Start the Voice Recorder (this blocks until you close it)
python recorder.py

:: 3. Cleanup: Once you close the recorder, kill the background window
echo Recorder closed. Stopping the background transcriber service...
taskkill /F /FI "WINDOWTITLE eq Audio Transcriber Background*" /T >nul 2>&1
echo All services completely shut down.
pause
