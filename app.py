import streamlit as st
import numpy as np
from scipy.io.wavfile import read
import soundfile as sf  # Add soundfile for better audio format support
import tempfile
import os
import time
from datetime import datetime
import whisper
import glob # To find files in the directory
import sys # Import sys to check if running in Streamlit
from enrich import enrich_with_llm_together, enrich_with_llm

# add load env
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Remove asyncio import as we won't use it

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
# Check if running in Streamlit before calling set_page_config
# This prevents errors if the script is run directly for testing functions
if 'streamlit' in sys.modules:
    st.set_page_config(layout="wide", page_title="Transcription App", page_icon=":material/speech_to_text:")

# --- Configuration ---
# Directory where the recorder script saves audio chunks.
# This MUST match the OUTPUT_DIR in recorder.py.
AUDIO_CHUNKS_DIR = "./audio_chunks"
TRANSCRIPTION_POLL_INTERVAL = 20 # Seconds to wait before checking for new files
OUTPUT_FILENAME = os.environ.get("TRANSCRIPT_PATH", "transcript_log.md") # File to save the transcript log
SUPPORTED_FORMATS = (".wav", ".mp3", "m4a")  # Add supported audio formats

# --- Load Whisper Model ---
# Use st.cache_resource to load the model only once across Streamlit re-runs.
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name=os.environ.get("WHISPER_MODEL", "turbo")):
    """Loads the Whisper model, cached to avoid reloading on re-runs."""
    try:
        with st.spinner(f"Loading Whisper model '{model_name}'..."):
            model = whisper.load_model(model_name)
            return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# Initialize model in session state if not already present
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = load_whisper_model()

# Use the model from session state
model = st.session_state.whisper_model

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




# --- UI Components ---
# Display the main title of the application
st.title(":material/speech_to_text: Transcription Monitor")
# Add a brief description
st.markdown(f"Monitoring `{AUDIO_CHUNKS_DIR}` for new audio files...")

# --- Main UI Layout ---
col1, col2 = st.columns(2)
with col1:
    # Start button. Disabled if the app is already processing.
    if st.button(":material/play_circle: Start Monitoring", type="primary", disabled=st.session_state.is_processing, use_container_width=True):
        st.session_state.is_processing = True # Set the state to processing
        st.session_state.status_message = f"Monitoring `{AUDIO_CHUNKS_DIR}`..."
        st.session_state.processed_files_count = 0
        st.session_state.error_count = 0
        st.rerun() # Force a Streamlit re-run to enter the processing loop

with col2:
    # Stop button. Disabled if the app is not processing.
    if st.button(":material/stop_circle: Stop Monitoring", type="secondary", disabled=not st.session_state.is_processing, use_container_width=True):
        st.session_state.is_processing = False # Set the state to stopping
        st.session_state.status_message = "Stopping monitoring..."
        st.rerun() # Force a Streamlit re-run to exit the processing loop

# Placeholders for dynamic UI elements
status_placeholder = st.empty() # Placeholder for status messages
progress_bar_placeholder = st.empty()
full_log_expander = st.container(border=False)  # Changed to expanded=True since it's now the only log
full_log_expander.subheader("Transcription Log", divider=True)
file_counts_placeholder = st.empty() # Placeholder to show file counts

# --- Audio Processing ---
def transcribe_audio_file(audio_path):
    """Transcribes an audio file (WAV or MP3) using the loaded Whisper model."""
    if model is None:
        return None, "Whisper model not loaded", None, None
    if not os.path.exists(audio_path):
        return None, f"Audio file not found: {audio_path}", None, None

    try:
        print(f"Transcribing audio: {audio_path}")

        # Get audio duration using soundfile
        audio_info = sf.info(audio_path)
        duration = audio_info.duration

        # Time the transcription process
        start_time = time.time()
        transcript = model.transcribe(audio_path)["text"].strip()
        
        processing_time = time.time() - start_time

        print(f"Transcription complete: {transcript[:50]}...")
        return transcript, None, duration, processing_time
    except Exception as e:
        print(f"Transcription error for {audio_path}: {e}")
        return None, str(e), None, None
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Removed processed file: {audio_path}")
            except OSError as e:
                print(f"Error removing processed file {audio_path}: {e}")


def append_to_markdown(text, filename):
    """Appends transcribed text to a markdown log file."""
    # Only append if there is valid text to write
    if text and text.strip():  # Check if text is not None or empty/whitespace
        try:
            # check if folder for the file exists, if not create it
            # Open the file in append mode ('a'). Creates the file if it doesn't exist.
            # Use utf-8 encoding for broad character support.
            with open(filename, "a", encoding="utf-8") as f:
                # Format the output with a timestamp and include duration and processing time
                f.write(text)
            print(f"Appended to log file: {filename}")  # Debugging
        except Exception as e:
            st.error(f"Error writing to log file: {e}")
            print(f"Error writing to log file: {e}")  # Debugging

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

        # Find all WAV and MP3 files in the audio chunks directory
        # Use glob to get a list of files matching the pattern
        # Sort the files by name (timestamp) to process them in recording order
        audio_files = []
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(glob.glob(os.path.join(AUDIO_CHUNKS_DIR, f"*{ext}")))
        audio_files.sort()  # Sort files by name

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

                transcript, error, duration, processing_time = transcribe_audio_file(file_path)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if transcript:
                    if os.getenv("ACTIVE_AI_ENRICHMENT", "True") == "True":
                        start_time = time.time()
                        llm_enriched_response = enrich_with_llm(transcript)
                        enrichment_time = time.time() - start_time
                        processing_time += enrichment_time   
                    else:
                        llm_enriched_response = None
                    
                    duration_str = f"**Audio Length:** {duration:.2f}s" if duration is not None else "**Audio Length:** unknown"
                    proc_time_str = f"**Processing Time:** {processing_time:.2f}s" if processing_time is not None else "**Processing Time:** unknown"
                    if llm_enriched_response:
                        summary_title = llm_enriched_response.get("title", "Untitled")
                        summary = llm_enriched_response.get("summary", "No summary available")
                        enriched_transcript = llm_enriched_response.get("speaker_enriched_transcript", transcript)
                        keywords = llm_enriched_response.get("keywords", [])
                        keywords_str = " ".join([f":blue-badge[{keyword}]" for keyword in keywords])
                        entry = f"#### [{timestamp}] {summary_title}\n\n**Topics:** {keywords_str}   |   {duration_str}   |   {proc_time_str}\n\n**Summary:** {summary}\n\n**Transcript:**\n\n{enriched_transcript}\n\n"
                    else:
                        entry = f"#### [{timestamp}] Title Unavailable   |   {duration_str}   |   {proc_time_str}\n\n**Transcript:**{transcript}\n\n"
                    st.session_state.transcript_log.insert(0, entry) # Add to log
                    append_to_markdown(entry, OUTPUT_FILENAME) # Append to file
                    st.session_state.processed_files_count += 1
                    print(f"Successfully processed {file_name}") # Debugging
                else:
                    entry = f"**{timestamp}:** ERROR transcribing {file_name} - {error}"
                    st.session_state.transcript_log.insert(0, entry) # Add error to log
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
        for entry in st.session_state.transcript_log[:30]:
            full_log_expander.markdown(entry.split("**Transcript:**")[0].strip())
            transcript_expander = full_log_expander.expander(label="**Transcript**")
            transcript_expander.markdown(entry.split("**Transcript:**")[1].strip())
        

        # --- Poll for New Files ---
        # Wait for a short interval before the next Streamlit re-run to check for new files.
        # This prevents the app from consuming excessive CPU by constantly polling.
        time.sleep(TRANSCRIPTION_POLL_INTERVAL)
        st.rerun() # Force a Streamlit re-run to check for new files again

    else:
        # This block executes on each Streamlit re-run *while* st.session_state.is_processing is False.
        status_placeholder.info(st.session_state.status_message)
        progress_bar_placeholder.progress(0.0, text="Monitoring stopped.")
        file_counts_placeholder.empty() # Clear file counts when stopped

        # Update the full log display one last time after stopping.
        for entry in st.session_state.transcript_log[:30]:
            full_log_expander.markdown(entry.split("**Transcript:**")[0].strip())
            transcript_expander = full_log_expander.expander(label="**Transcript**")
            transcript_expander.markdown(entry.split("**Transcript:**")[1].strip())
            
        # full_log_expander.markdown("\n\n---\n\n".join(st.session_state.transcript_log[:30]))

# --- Script Entry Point ---
# This is where the Streamlit script execution begins on every re-run.
# It simply calls the main application logic function.
if __name__ == "__main__":
    main_loop() # Start the main Streamlit loop
