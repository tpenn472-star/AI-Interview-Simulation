# Evalify AI Interview Service - Clean Final Version

This project contains only the final application flow for the AI interview simulation backend.

## Active endpoints

### 1. Generate interview questions

`POST /ai/interview/questions`

### 2. Generate question audio

`POST /ai/interview/question-tts`

### 3. Analyze full interview session audio

`POST /ai/interview/analyze-session-audio-batch`

This endpoint performs:

1. Audio conversion to WAV 16 kHz
2. Speech-to-text using Whisper
3. Binary filler detection using TensorFlow
4. Batch GenAI evaluation
5. Final scoring

## Required files

Create `.env` from `.env.example`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
WHISPER_MODEL_SIZE=base
```

Model files must be placed in `models/`:

```text
models/best_podcastfillers_filler_detector.keras
models/podcastfillers_filler_detector_metadata.json
```

## Run

```powershell
python -m uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Removed from this clean version

- Debug endpoints
- Single answer evaluation endpoint
- Single audio analysis endpoint
- Standalone transcription endpoint
- Standalone filler-detection endpoint
- Old uploaded audio files
- Generated TTS files
- `__pycache__`
- local `.env`
- unused `q.py`
