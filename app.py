import streamlit as st
import numpy as np
from scipy.io.wavfile import read # Use read to load WAV files
import tempfile
import os
import time
from datetime import datetime
import whisper
import glob # To find files in the directory
import sys # Import sys to check if running in Streamlit

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
# Check if running in Streamlit before calling set_page_config
# This prevents errors if the script is run directly for testing functions
if 'streamlit' in sys.modules:
    st.set_page_config(layout="wide", page_title="Whisper Transcription")

# --- Configuration ---
# Directory where the recorder script saves audio chunks.
# This MUST match the OUTPUT_DIR in recorder.py.
AUDIO_CHUNKS_DIR = "./audio_chunks"
TRANSCRIPTION_POLL_INTERVAL = 10 # Seconds to wait before checking for new files
OUTPUT_FILENAME = "transcript_log.md" # File to save the transcript log

# --- Load Whisper Model ---
# Use st.cache_resource to load the model only once across Streamlit re-runs.
@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model, cached to avoid reloading on re-runs."""
    try:
        # Using the 'base' model. Can be changed to 'tiny', 'small', 'medium', 'large'.
        # 'base' is a good balance for general use.
        model_name = "medium" # Change to 'tiny', 'small', 'medium', or 'large' as needed
        # Use st.spinner to show a loading indicator while the model is loading.
        with st.spinner(f"Loading Whisper model '{model_name}'..."):
             model = whisper.load_model(model_name)
        st.success(f"Whisper model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# Load the model when the script first runs
model = load_whisper_model()

# --- Session State Initialization ---
# Initialize necessary session state variables.
# Session state persists across Streamlit re-runs.
required_states = [
    'transcript_log',      # List to store transcript entries displayed in the UI
    'is_processing',       # Boolean flag: True when the app is actively monitoring/transcribing
    'status_message',      # Message displayed in the status placeholder
    'processed_files_count', # Counter for successfully processed files
    'error_count'          # Counter for files that failed transcription
]

# Loop through required states and initialize if they don't exist.
for state in required_states:
    if state not in st.session_state:
        if state == 'transcript_log':
            st.session_state[state] = [] # Initialize transcript log as an empty list
        elif state == 'is_processing':
            st.session_state[state] = False # App starts in stopped state
        elif state == 'status_message':
            st.session_state[state] = "Ready to start monitoring." # Initial status
        elif state in ['processed_files_count', 'error_count']:
             st.session_state[state] = 0 # Counters initialized to zero


# --- Audio Processing ---
def transcribe_wav_file(wav_path):
    """Transcribes a WAV file using the loaded Whisper model."""
    if model is None:
        # If the model failed to load earlier, return an error
        return None, "Whisper model not loaded"
    if not os.path.exists(wav_path):
         # If the WAV file doesn't exist, return an error
         return None, f"WAV file not found: {wav_path}"

    try:
        print(f"Transcribing WAV: {wav_path}") # Debugging
        # Load the audio file using scipy.io.wavfile.read
        # read returns sample rate and data
        sample_rate, audio_data = read(wav_path)

        # Ensure audio data is in the correct format (int16) and mono if necessary
        if audio_data.dtype != np.int16:
             # Convert to int16 if it's not already
             audio_data = (audio_data * (2**15 - 1)).astype(np.int16)

        if audio_data.ndim > 1:
             # Convert to mono by taking the first channel
             audio_data = audio_data[:, 0]

        # Whisper expects a specific format (float32, 16kHz).
        # We need to resample if the input is not 16kHz.
        # For simplicity in this example, we assume 16kHz input from recorder.py.
        # If your recorder saves at a different rate, you'd need resampling here (e.g., using librosa or torchaudio).
        if sample_rate != 16000:
             print(f"Warning: Input sample rate {sample_rate} != 16000. Transcription quality may be affected.")
             # Add resampling logic here if needed

        # Whisper's transcribe method expects a numpy array of type float32
        # and a sample rate of 16000. The base model works with 16kHz.
        # Convert int16 audio data to float32 and normalize
        audio_data_float32 = audio_data.astype(np.float32) / (2**15)


        # Perform the transcription using the Whisper model.
        # The transcribe method returns a dictionary, we extract the 'text' key.
        transcript = model.transcribe(audio_data_float32)["text"].strip()
        print(f"Transcription complete: {transcript[:50]}...") # Debugging (print first 50 chars)
        return transcript, None # Return the transcribed text and no error
    except Exception as e:
        # Catch any errors during transcription (e.g., model issues, input format)
        print(f"Transcription error for {wav_path}: {e}") # Debugging
        return None, str(e) # Return None for transcript and the error message as a string
    finally:
        # This block ensures the temporary file is removed after processing.
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path) # Attempt to delete the processed file
                print(f"Removed processed WAV: {wav_path}") # Debugging
            except OSError as e:
                 # Handle potential issues if the file is still in use or deletion fails
                 print(f"Error removing processed file {wav_path}: {e}")
                 # In a production app, you might want more robust error handling or logging here.


def append_to_markdown(text, filename):
    """Appends transcribed text to a markdown log file."""
    # Only append if there is valid text to write
    if text and text.strip(): # Check if text is not None or empty/whitespace
        try:
            # Open the file in append mode ('a'). Creates the file if it doesn't exist.
            # Use utf-8 encoding for broad character support.
            with open(filename, "a", encoding="utf-8") as f:
                # Format the output with a timestamp and markdown heading/paragraph structure
                f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n\n")
            print(f"Appended to log file: {filename}") # Debugging
        except Exception as e:
            st.error(f"Error writing to log file: {e}")
            print(f"Error writing to log file: {e}") # Debugging

# --- UI Components ---
# Display the main title of the application
st.title("🎙️ Continuous Transcription Monitor")
# Add a brief description
st.markdown(f"Monitoring `{AUDIO_CHUNKS_DIR}` for new audio files...")

# --- Main UI Layout ---
col1, col2 = st.columns(2)
with col1:
    # Start button. Disabled if the app is already processing.
    if st.button("▶️ Start Monitoring", type="primary", disabled=st.session_state.is_processing):
        st.session_state.is_processing = True # Set the state to processing
        st.session_state.status_message = f"Monitoring `{AUDIO_CHUNKS_DIR}`..."
        st.session_state.processed_files_count = 0
        st.session_state.error_count = 0
        st.rerun() # Force a Streamlit re-run to enter the processing loop

with col2:
    # Stop button. Disabled if the app is not processing.
    if st.button("⏹️ Stop Monitoring", type="secondary", disabled=not st.session_state.is_processing):
        st.session_state.is_processing = False # Set the state to stopping
        st.session_state.status_message = "Stopping monitoring..."
        st.rerun() # Force a Streamlit re-run to exit the processing loop

# Placeholders for dynamic UI elements
status_placeholder = st.empty() # Placeholder for status messages
# Progress bar could show files processed vs total found in a batch, or just general activity
progress_bar_placeholder = st.empty()
latest_transcript_expander = st.expander("Latest Transcript", expanded=True)
full_log_expander = st.expander("Full Log", expanded=False)
file_counts_placeholder = st.empty() # Placeholder to show file counts

# --- Main Application Logic ---
def main_loop():
    """
    The main Streamlit function that runs on each re-run.
    It monitors the audio directory, processes files, and updates the UI.
    """
    # Ensure the audio chunks directory exists (useful if recorder.py hasn't run yet)
    if not os.path.exists(AUDIO_CHUNKS_DIR):
         st.warning(f"Audio chunks directory not found: `{AUDIO_CHUNKS_DIR}`. Please run `recorder.py` first.")
         st.session_state.is_processing = False # Cannot process if directory doesn't exist
         st.session_state.status_message = f"Directory `{AUDIO_CHUNKS_DIR}` not found."


    # --- Monitoring and Processing Loop ---
    # This block executes on every Streamlit re-run *while* st.session_state.is_processing is True.
    if st.session_state.is_processing:
        status_placeholder.info(st.session_state.status_message)

        # Find all WAV files in the audio chunks directory
        # Use glob to get a list of files matching the pattern
        # Sort the files by name (timestamp) to process them in recording order
        audio_files = sorted(glob.glob(os.path.join(AUDIO_CHUNKS_DIR, "*.wav")))

        if not audio_files:
            st.session_state.status_message = f"Monitoring `{AUDIO_CHUNKS_DIR}`. No new files found."
            status_placeholder.info(st.session_state.status_message)
            progress_bar_placeholder.progress(0.0, text="Waiting for audio files...")
        else:
            st.session_state.status_message = f"Monitoring `{AUDIO_CHUNKS_DIR}`. Found {len(audio_files)} file(s) to process."
            status_placeholder.info(st.session_state.status_message)

            # Process files one by one
            for i, file_path in enumerate(audio_files):
                file_name = os.path.basename(file_path)
                progress_text = f"Processing file {i + 1}/{len(audio_files)}: {file_name}"
                progress_value = (i + 1) / len(audio_files)
                progress_bar_placeholder.progress(progress_value, text=progress_text)
                st.session_state.status_message = progress_text # Update status message too

                transcript, error = transcribe_wav_file(file_path)

                timestamp = datetime.now().strftime("%H:%M:%S")

                if transcript:
                    entry = f"**{timestamp}:** {transcript}"
                    st.session_state.transcript_log.insert(0, entry) # Add to log
                    latest_transcript_expander.markdown(entry) # Update latest display
                    append_to_markdown(transcript, OUTPUT_FILENAME) # Append to file
                    st.session_state.processed_files_count += 1
                    print(f"Successfully processed {file_name}") # Debugging
                else:
                    entry = f"**{timestamp}:** ERROR transcribing {file_name} - {error}"
                    st.session_state.transcript_log.insert(0, entry) # Add error to log
                    latest_transcript_expander.markdown(entry) # Update latest display
                    append_to_markdown(f"ERROR: {error} (File: {file_name})", OUTPUT_FILENAME) # Append error to file
                    st.session_state.error_count += 1
                    print(f"Error processing {file_name}: {error}") # Debugging

            # After processing all files in this batch:
            st.session_state.status_message = f"Finished processing batch. Processed {st.session_state.processed_files_count} files, {st.session_state.error_count} errors."
            status_placeholder.success(st.session_state.status_message)
            progress_bar_placeholder.progress(1.0, text="Batch processing complete.") # Set progress to 100%

        # Update file counts display
        file_counts_placeholder.info(
            f"Processed: {st.session_state.processed_files_count} | Errors: {st.session_state.error_count}"
        )

        # Update the "Full Log" expander display with the latest entries.
        # Limiting the number of displayed entries (`[:30]`) can improve performance
        # for very long transcripts.
        full_log_expander.markdown("\n\n---\n\n".join(st.session_state.transcript_log[:30]))


        # --- Poll for New Files ---
        # Wait for a short interval before the next Streamlit re-run to check for new files.
        # This prevents the app from consuming excessive CPU by constantly polling.
        time.sleep(TRANSCRIPTION_POLL_INTERVAL)
        st.rerun() # Force a Streamlit re-run to check for new files again

    else:
        # This block executes on each Streamlit re-run *while* st.session_state.is_processing is False.
        # It ensures the UI reflects the stopped state.
        status_placeholder.info(st.session_state.status_message)
        progress_bar_placeholder.progress(0.0, text="Monitoring stopped.")
        file_counts_placeholder.empty() # Clear file counts when stopped

        # Update the full log display one last time after stopping.
        full_log_expander.markdown("\n\n---\n\n".join(st.session_state.transcript_log[:30]))


# --- Script Entry Point ---
# This is where the Streamlit script execution begins on every re-run.
# It simply calls the main application logic function.
if __name__ == "__main__":
    main_loop() # Start the main Streamlit loop
