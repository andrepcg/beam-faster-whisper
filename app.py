from beam import endpoint, Image, Volume, env
import base64
import requests
from tempfile import NamedTemporaryFile

from predict import Predictor
from models import load_models, cache_path

def parse_audio(audio_base64, url):
    if audio_base64 and url:
        throw("Only a base64 audio file OR a URL can be passed to the API.")
        # return {"error": "Only a base64 audio file OR a URL can be passed to the API."}
    if not audio_base64 and not url:
        throw("Please provide either an audio file in base64 string format or a URL.")
        # return {
        #     "error": "Please provide either an audio file in base64 string format or a URL."
        # }

    binary_data = None

    if audio_base64:
        binary_data = base64.b64decode(audio_base64.encode("utf-8"))
    elif url:
        resp = requests.get(url)
        binary_data = resp.content

    with NamedTemporaryFile() as temp:
        # Write the audio data to the temporary file
        temp.write(binary_data)
        temp.flush()

        return temp


def on_start():
    print("On startup running")
    return load_models()

@endpoint(
    on_start=on_start,
    name="faster-whisper",
    cpu=1,
    memory="8Gi",
    gpu="T4",
    image=Image(
        base_image="nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        python_version="python3.10",
    )
    .add_python_packages("./requirements.txt")
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    volumes=[
        Volume(name="cached_models", mount_path=cache_path)
    ],
)
def transcribe(context, **inputs):
    # Retrieve cached model from on_start
    models = context.on_start_value

    # Inputs passed to API
    language = inputs.get("language")
    audio_base64 = inputs.get("audio_file")
    transcription = inputs.get("transcription")
    url = inputs.get("url")
    model_name = inputs.get("model", "large-v3-turbo")

    model = models.get(model_name)
    if not model:
        return {"error": f"Model {model_name} not found."}

    text = ""

    try:
        temp_file = parse_audio(audio_base64, url)
        # results = Predictor(model).predict(temp.name, transcription=transcription, language=language)
        # return results

        segments, _ = model.transcribe(temp_file.name, beam_size=5, language=language)

        for segment in segments:
            text += segment.text + " "

        print(text)
        return {"text": text}

    except Exception as e:
        return {"error": f"Something went wrong: {e}"}
