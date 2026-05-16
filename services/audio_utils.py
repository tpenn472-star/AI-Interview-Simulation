import shutil
import subprocess
from pathlib import Path

from fastapi import HTTPException, UploadFile


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination


def convert_to_wav_16k(input_path: Path) -> Path:
    output_path = input_path.with_suffix(".16k.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(output_path)
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg was not found. Install ffmpeg and add it to PATH.")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert audio: {e.stderr.decode(errors='ignore')}")

    return output_path
