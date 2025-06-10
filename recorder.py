import soundcard as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
import threading
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
CHUNK_DURATION = int(os.getenv("RECORDER_CHUNK_DURATION", 300))  # Duration of each audio chunk in seconds
SAMPLE_RATE = 16000  # Sample rate for audio recording
CHANNELS = 1         # Number of audio channels (1 for mono)
# Directory to save audio chunks. Make sure this directory exists before running.
# A fixed directory is used here for simplicity, ensure both scripts can access it.
OUTPUT_DIR = "./audio_chunks"

mic = sd.all_microphones(include_loopback=True)[os.getenv("RECORDER_MIC_INDEX", 0)]  # Select the first available microphone
# --- Global Flag ---
is_recording = False # Flag to control the recording loop

# --- Ensure Output Directory Exists ---
def ensure_output_dir():
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    else:
        print(f"Output directory already exists: {OUTPUT_DIR}")

# --- Recording Function ---
def record_chunk():
    """Records a single audio chunk and saves it to a file."""
    global is_recording # Moved this to be the absolute first line
    try:
        chunk_size = int(CHUNK_DURATION * SAMPLE_RATE)
        print(f"Recording chunk of {CHUNK_DURATION} seconds...")

        # Record audio using sounddevice. This call blocks until the recording is complete.
        recording = mic.record(numframes=chunk_size, samplerate=SAMPLE_RATE)
        sd.wait() # Wait until the recording is complete

        if is_recording: # This is the usage that was reported as being before the global declaration
            # Create a unique filename using a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(OUTPUT_DIR, f"audio_chunk_{timestamp}.wav")

            # Save the recorded data to a WAV file
            write(filename, SAMPLE_RATE, recording)
            print(f"Saved chunk to {filename}")
            return filename # Return the saved filename
        else:
            print("Recording stopped during chunk capture, not saving.")
            return None

    except Exception as e:
        print(f"Recording error: {e}")
        # In a real application, you might want more sophisticated error handling,
        # like logging the error and attempting to restart recording.
        # global is_recording # Removed redundant global declaration here
        is_recording = False # Stop the recording loop on error
        return None

# --- Main Recording Loop ---
def continuous_recording_loop():
    """Continuously records audio chunks until the stop flag is set."""
    global is_recording
    print("Continuous recording loop started.")
    ensure_output_dir() # Ensure the output directory exists

    while is_recording:
        record_chunk()
        # No need for a sleep here, sd.wait() handles the chunk duration.
        # The loop will proceed to the next chunk immediately after saving.

    print("Continuous recording loop stopped.")

# --- Script Entry Point ---
if __name__ == "__main__":
    print("Starting audio recorder script.")
    # You would typically start and stop this script manually or via a service.
    # For demonstration, we'll set is_recording to True and run the loop.
    # To stop, you would need to terminate the script (e.g., Ctrl+C).

    is_recording = True # Set the flag to start recording

    try:
        continuous_recording_loop()
    except KeyboardInterrupt:
        print("\nRecording stopped by user (KeyboardInterrupt).")
        is_recording = False # Ensure the flag is set to False
        # The loop will exit after the current chunk finishes.

    print("Audio recorder script finished.")
