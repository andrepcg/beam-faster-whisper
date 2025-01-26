import subprocess


def is_cuda_available():
    """
    Returns True if CUDA is available, False otherwise.
    """
    try:
        output = subprocess.check_output("nvidia-smi", shell=True)
        if "NVIDIA-SMI" in output.decode():
            return True
    except Exception:  # pylint: disable=broad-except
        pass
    return False
