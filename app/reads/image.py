import os


def read_image_bytes(file_path: str) -> bytes:
    """Read an image file and return its raw bytes for vision models."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'rb') as f:
        return f.read()
