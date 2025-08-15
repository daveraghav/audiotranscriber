import os
import time
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)


def _log(msg: str, verbose: bool = False):
    """Print a timestamped message when verbose is enabled."""
    if not verbose:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def process_audio_file(file_path, deepgram_client):
    with open(file_path, "rb") as audio_file:
        buffer_data = audio_file.read()
    payload: FileSource = {
        "buffer": buffer_data,
    }
    options = PrerecordedOptions(
        model="nova-3",
        language="multi",
        smart_format=True,
        paragraphs=True,
        diarize=True,
    )
    response = deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
    now = datetime.now()
    output_dir = os.path.join(
        "transcripts",
        "json_data",
        str(now.year),
        f"{now.month:02d}",
        f"{now.day:02d}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_filename = now.strftime("%H-%M.json")
    output_path = os.path.join(output_dir, output_filename)

    def make_serializable(obj):
        """Recursively convert an object into JSON-serializable primitives.

        Handles dicts, lists, tuples, sets, bytes, and objects that provide
        convenient conversion methods (to_dict, to_json, json, dict). Falls
        back to string() for unknown types.
        """
        # primitives
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # bytes -> try decode, else list of ints
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8")
            except Exception:
                return list(obj)

        # dict
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}

        # iterables
        if isinstance(obj, (list, tuple, set)):
            return [make_serializable(v) for v in obj]

        # common library helpers
        for attr in ("to_dict", "dict", "to_json", "json", "toJSON"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    result = fn()
                    # if result is a JSON string, try to load it
                    if isinstance(result, str):
                        try:
                            return json.loads(result)
                        except Exception:
                            return result
                    return make_serializable(result)
                except Exception:
                    # fall through and try other strategies
                    pass

        # dataclass / objects with __dict__
        if hasattr(obj, "__dict__"):
            try:
                return make_serializable(vars(obj))
            except Exception:
                pass

        # final fallback
        return str(obj)

    serializable = make_serializable(response)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Watch audio_chunks and transcribe new audio files.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose timestamped logging")
    args = parser.parse_args()

    verbose = args.verbose
    audio_chunks_dir = "audio_chunks"

    # ensure the audio_chunks directory exists
    os.makedirs(audio_chunks_dir, exist_ok=True)
    _log(f"Watching directory: {audio_chunks_dir}", verbose)

    while True:
        try:
            entries = os.listdir(audio_chunks_dir)
        except Exception as e:
            _log(f"Failed to list '{audio_chunks_dir}': {e}", True)
            time.sleep(10)
            continue

        audio_files = [f for f in entries if f.lower().endswith((".wav", ".mp3"))]

        if not audio_files:
            _log(f"No audio files found in '{audio_chunks_dir}'; waiting for new files...", verbose)
            time.sleep(10)
            continue

        for filename in audio_files:
            file_path = os.path.join(audio_chunks_dir, filename)
            _log(f"Starting processing: {file_path}", verbose)
            try:
                process_audio_file(file_path, deepgram_client)
                _log(f"Finished processing: {file_path}", verbose)
            except Exception as e:
                # always print errors
                _log(f"Error processing {file_path}: {e}", True)
            try:
                os.remove(file_path)
            except Exception as e:
                _log(f"Failed to remove {file_path}: {e}", True)

        # After processing current batch, wait before checking again
        _log(f"Waiting for new files to arrive in '{audio_chunks_dir}'...", verbose)
        time.sleep(10)

if __name__ == "__main__":
    main()
