from models import load_models, cache_path
import os
import argparse

models = load_models()

# use command line arguments to pass the audio file and model to use


parser = argparse.ArgumentParser(description="Transcribe an audio file.")
parser.add_argument("audio_file", type=str, help="Path to the audio file.")
parser.add_argument("--model", type=str, default="large-v3-turbo", help="Model to use.")
parser.add_argument("--language", type=str, default="en", help="Language code.")

args = parser.parse_args()

model = models.get(args.model)
if not model:
    print(f"Model {args.model} not found.")
    exit(1)

audio_file = args.audio_file
if not os.path.exists(audio_file):
    print(f"Audio file {audio_file} not found.")
    exit(1)

segments, _ = model.transcribe(audio_file, beam_size=5)

for segment in segments:
    print(segment.text)
