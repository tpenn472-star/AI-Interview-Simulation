from fastapi import APIRouter

from constants import EVALIFY_RUBRIC


router = APIRouter()


@router.get("/")
def root():
    return {
        "service": "Evalify AI Interview Service",
        "status": "running",
        "mode": "final_application_flow",
        "final_flow": [
            "1. POST /ai/interview/questions",
            "2. POST /ai/interview/question-tts",
            "3. POST /ai/interview/analyze-session-audio-batch"
        ]
    }


@router.get("/ai/interview/rubric")
def get_rubric():
    return {
        "rubric": EVALIFY_RUBRIC,
        "total_weight": sum(item["weight"] for item in EVALIFY_RUBRIC.values())
    }
