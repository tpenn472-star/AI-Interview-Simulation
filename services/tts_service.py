import uuid
from pathlib import Path

import pyttsx3
from fastapi import HTTPException

from config import TTS_DIR


def generate_question_tts(text: str) -> Path:
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")

    audio_id = str(uuid.uuid4())
    output_path = TTS_DIR / f"question_{audio_id}.wav"

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate TTS audio: {e}")

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="TTS audio file was not created.")

    return output_path
