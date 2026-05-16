from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from schemas import GenerateQuestionsRequest, GenerateQuestionsResponse
from services.genai_service import generate_questions_service
from services.orchestrator_service import analyze_session_audio_batch_core
from services.tts_service import generate_question_tts


router = APIRouter(prefix="/ai/interview")


@router.post("/questions", response_model=GenerateQuestionsResponse)
def generate_questions(req: GenerateQuestionsRequest):
    return generate_questions_service(req)


@router.post("/question-tts")
def question_tts(
    text: str = Form(...),
    language: str = Form("en")
):
    output_path = generate_question_tts(text)

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=output_path.name
    )


@router.post("/analyze-session-audio-batch")
def analyze_session_audio_batch(
    role: str = Form(...),
    experience_level: str = Form("fresh graduate"),
    language: str = Form("en"),
    answers_json: str = Form(...),
    audio_1: UploadFile = File(...),
    audio_2: Optional[UploadFile] = File(None),
    audio_3: Optional[UploadFile] = File(None),
    audio_4: Optional[UploadFile] = File(None),
    audio_5: Optional[UploadFile] = File(None),
    audio_6: Optional[UploadFile] = File(None),
    audio_7: Optional[UploadFile] = File(None),
    audio_8: Optional[UploadFile] = File(None),
    audio_9: Optional[UploadFile] = File(None),
    audio_10: Optional[UploadFile] = File(None),
    audio_11: Optional[UploadFile] = File(None),
    audio_12: Optional[UploadFile] = File(None)
):
    audio_files = [
        audio_1,
        audio_2,
        audio_3,
        audio_4,
        audio_5,
        audio_6,
        audio_7,
        audio_8,
        audio_9,
        audio_10,
        audio_11,
        audio_12,
    ]
    audio_files = [audio for audio in audio_files if audio is not None]

    return analyze_session_audio_batch_core(
        role=role,
        experience_level=experience_level,
        language=language,
        answers_json=answers_json,
        audio_files=audio_files
    )
