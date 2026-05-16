import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TTS_DIR = BASE_DIR / "generated_tts"
MODEL_DIR = BASE_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
TTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FILLER_MODEL_PATH = MODEL_DIR / "best_podcastfillers_filler_detector.keras"
FILLER_METADATA_PATH = MODEL_DIR / "podcastfillers_filler_detector_metadata.json"
FILLER_DECISION_THRESHOLD = 0.50
