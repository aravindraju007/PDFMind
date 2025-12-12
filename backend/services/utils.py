import os
from fastapi import UploadFile
from typing import Optional

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(file: UploadFile, prefix: Optional[str] = "") -> str:
    """
    Save FastAPI UploadFile to disk and return saved path.
    """
    filename = prefix + file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        contents = file.file.read()
        f.write(contents)
    # rewind file pointer in case caller needs it
    try:
        file.file.seek(0)
    except Exception:
        pass
    return path
