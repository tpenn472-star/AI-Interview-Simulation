import json
import uuid
from typing import List

from fastapi import HTTPException, UploadFile

from config import UPLOAD_DIR
from schemas import BatchAnswerItem, EvaluateSessionBatchRequest
from services.audio_utils import convert_to_wav_16k, save_upload_file
from services.filler_service import analyze_filler_from_wav
from services.genai_service import evaluate_session_batch_service
from services.scoring_service import combine_content_delivery_score
from services.stt_service import transcribe_wav


def analyze_session_audio_batch_core(
    role: str,
    experience_level: str,
    language: str,
    answers_json: str,
    audio_files: List[UploadFile]
):
    try:
        answer_items = json.loads(answers_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"answers_json must be valid JSON: {e}")

    if not isinstance(answer_items, list) or not answer_items:
        raise HTTPException(status_code=400, detail="answers_json must be a non-empty JSON array.")

    if len(answer_items) != len(audio_files):
        raise HTTPException(
            status_code=400,
            detail=f"answers_json length ({len(answer_items)}) must match audio_files length ({len(audio_files)})."
        )

    processed_answers = []
    delivery_results = []

    for idx, (item, audio) in enumerate(zip(answer_items, audio_files), start=1):
        question_id = str(item.get("question_id", f"q_{idx}"))
        interview_type = str(item.get("interview_type", "general"))
        question = str(item.get("question", "")).strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail=f"question must not be empty for item index {idx}."
            )

        raw_path = UPLOAD_DIR / f"{uuid.uuid4()}_{audio.filename}"
        save_upload_file(audio, raw_path)

        wav_path = convert_to_wav_16k(raw_path)

        stt_result = transcribe_wav(wav_path)
        transcript = stt_result.get("transcript", "").strip()

        if not transcript:
            transcript = "[No clear transcript detected from the audio.]"

        filler_result = analyze_filler_from_wav(wav_path)

        processed_answers.append(
            BatchAnswerItem(
                question_id=question_id,
                interview_type=interview_type,
                question=question,
                answer_text=transcript
            )
        )

        delivery_results.append({
            "question_id": question_id,
            "question": question,
            "interview_type": interview_type,
            "audio_filename": audio.filename,
            "transcript": transcript,
            "stt": stt_result,
            "speech_delivery": filler_result,
            "delivery_score": float(filler_result.get("delivery_score", 0))
        })

    batch_req = EvaluateSessionBatchRequest(
        role=role,
        experience_level=experience_level,
        language=language,
        answers=processed_answers
    )

    content_batch_result = evaluate_session_batch_service(batch_req)

    content_results = content_batch_result.get("results", [])
    content_by_question_id = {
        str(item.get("question_id")): item
        for item in content_results
    }

    combined_results = []

    for delivery_item in delivery_results:
        question_id = delivery_item["question_id"]
        content_item = content_by_question_id.get(question_id, {})

        content_score = float(content_item.get("overall_score", 0))
        delivery_score = float(delivery_item.get("delivery_score", 0))

        combined_results.append({
            "question_id": question_id,
            "interview_type": delivery_item["interview_type"],
            "question": delivery_item["question"],
            "audio_filename": delivery_item["audio_filename"],
            "transcript": delivery_item["transcript"],
            "content_evaluation": content_item,
            "speech_delivery": delivery_item["speech_delivery"],
            "score_breakdown": combine_content_delivery_score(content_score, delivery_score)
        })

    average_content_score = content_batch_result.get("average_content_score", 0)

    average_delivery_score = sum(
        item["score_breakdown"]["delivery_score"] for item in combined_results
    ) / max(len(combined_results), 1)

    average_final_score = sum(
        item["score_breakdown"]["final_score"] for item in combined_results
    ) / max(len(combined_results), 1)

    return {
        "role": role,
        "experience_level": experience_level,
        "language": language,
        "total_answers": len(combined_results),
        "genai_call_count_for_content_evaluation": content_batch_result.get("genai_call_count_for_this_endpoint"),
        "max_answers_per_genai_call": content_batch_result.get("max_answers_per_genai_call"),
        "average_content_score": round(float(average_content_score), 2),
        "average_delivery_score": round(float(average_delivery_score), 2),
        "average_final_score": round(float(average_final_score), 2),
        "final_session_feedback": content_batch_result.get("final_session_feedback", {}),
        "results": combined_results
    }
