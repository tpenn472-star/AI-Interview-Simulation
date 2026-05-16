from pathlib import Path
from typing import Any, Dict

from faster_whisper import WhisperModel

from config import WHISPER_MODEL_SIZE


_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

    return _whisper_model


def transcribe_wav(audio_path: Path) -> Dict[str, Any]:
    model = get_whisper_model()

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=None
    )

    transcript_parts = []
    segment_list = []

    for seg in segments:
        text = seg.text.strip()
        transcript_parts.append(text)
        segment_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text
        })

    transcript = " ".join(transcript_parts).strip()

    return {
        "transcript": transcript,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segment_list
    }
