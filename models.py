
from utils import is_cuda_available

BEAM_VOLUME_PATH = "./cached_models"

def cache_path():
    return BEAM_VOLUME_PATH

def build_model(path):
    from faster_whisper import WhisperModel

    device = "cuda" if is_cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(path, device=device, compute_type=compute_type)

# This runs once when the container first starts
def load_models():
    from faster_whisper import WhisperModel, download_model
    from huggingface_hub import snapshot_download

    print("Loading models...")

    model_large = build_model(download_model("large-v3", cache_dir=BEAM_VOLUME_PATH))

    model_path = snapshot_download(repo_id="deepdml/faster-whisper-large-v3-turbo-ct2", repo_type="model", cache_dir=BEAM_VOLUME_PATH)
    model_large_turbo = build_model(model_path)

    models = {"large-v3": model_large, "large-v3-turbo": model_large_turbo}

    print("Models loaded:")
    for key in models:
        print(key)

    return models