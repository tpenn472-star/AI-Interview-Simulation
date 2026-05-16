import json
import re
import uuid
from typing import Any, Dict, List

from fastapi import HTTPException
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL
from constants import EVALIFY_RUBRIC, QUESTION_TYPES
from schemas import EvaluateSessionBatchRequest, GenerateQuestionsRequest, GenerateQuestionsResponse, InterviewQuestion
from services.scoring_service import build_rule_based_final_feedback, score_to_grade


genai_client = genai.Client(api_key=GEMINI_API_KEY)


def extract_json(text: str) -> Any:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            return json.loads(text[start_obj:end_obj + 1])

        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            return json.loads(text[start_arr:end_arr + 1])

        raise


def call_gemini_json(prompt: str) -> Any:
    response = genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        )
    )
    return extract_json(response.text)


def build_question_prompt(req: GenerateQuestionsRequest, selected_types: List[str]) -> str:
    selected_type_descriptions = {
        qtype: QUESTION_TYPES[qtype]
        for qtype in selected_types
    }

    profile = req.user_profile_summary or "No additional candidate profile was provided."

    return f"""
You are an AI interviewer for the Evalify interview simulation application.

Selected role:
{req.role}

Candidate level:
{req.experience_level}

Candidate profile:
{profile}

Question types to generate:
{json.dumps(selected_type_descriptions, ensure_ascii=False, indent=2)}

Number of questions per type:
{req.num_questions_per_type}

Instructions:
1. Generate realistic job interview questions based on the selected role.
2. Every requested question type must be included.
3. Technical and role-specific questions must be relevant to the selected role.
4. Behavioral questions should be answerable using the STAR method.
5. Avoid discriminatory, sensitive, political, health-related, family-status, religious, or ethnicity-related questions.
6. Use professional and natural English.
7. Output valid JSON only.

Output format:
{{
  "questions": [
    {{
      "interview_type": "hr",
      "question": "string",
      "purpose": "string",
      "expected_answer_points": ["string", "string", "string"]
    }}
  ]
}}
"""


def generate_questions_service(req: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    selected_types = req.question_types or list(QUESTION_TYPES.keys())
    selected_types = [
        qtype for qtype in selected_types
        if qtype in QUESTION_TYPES
    ]

    if not selected_types:
        selected_types = list(QUESTION_TYPES.keys())

    prompt = build_question_prompt(req, selected_types)

    try:
        data = call_gemini_json(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")

    questions_raw = data.get("questions", [])

    if not questions_raw:
        raise HTTPException(status_code=500, detail="GenAI did not return any questions.")

    questions = []

    for item in questions_raw:
        qtype = str(item.get("interview_type", "hr")).lower().strip()
        if qtype not in QUESTION_TYPES:
            qtype = "hr"

        questions.append(
            InterviewQuestion(
                question_id=str(uuid.uuid4()),
                interview_type=qtype,
                question=str(item.get("question", "")).strip(),
                purpose=str(item.get("purpose", "")).strip(),
                expected_answer_points=[
                    str(x).strip()
                    for x in item.get("expected_answer_points", [])
                    if str(x).strip()
                ],
            )
        )

    return GenerateQuestionsResponse(
        session_id=str(uuid.uuid4()),
        role=req.role,
        experience_level=req.experience_level,
        language=req.language,
        questions=questions,
        rubric=EVALIFY_RUBRIC,
    )


def build_batch_evaluation_prompt(req: EvaluateSessionBatchRequest) -> str:
    answers_payload = [
        {
            "question_id": item.question_id,
            "interview_type": item.interview_type,
            "question": item.question,
            "answer_text": item.answer_text,
        }
        for item in req.answers
    ]

    return f"""
You are an AI interview evaluator for the Evalify interview simulation application.

Role:
{req.role}

Candidate level:
{req.experience_level}

Evaluation rubric:
{json.dumps(EVALIFY_RUBRIC, ensure_ascii=False, indent=2)}

Candidate answers:
{json.dumps(answers_payload, ensure_ascii=False, indent=2)}

Instructions:
1. Evaluate every answer independently based on the rubric.
2. Give each rubric aspect a score from 0 to 100.
3. overall_score must be the weighted final score based on the rubric weights.
4. Do not evaluate audio quality, filler words, or speaking delivery here. Those are handled by the filler detection module.
5. If an answer is too short, too generic, or does not answer the question, the score must be low.
6. Provide specific and actionable feedback for each answer.
7. After evaluating all answers, provide final session feedback.
8. Use professional and natural English.
9. Output valid JSON only.

Output format:
{{
  "average_content_score": 0,
  "results": [
    {{
      "question_id": "string",
      "interview_type": "string",
      "question": "string",
      "overall_score": 0,
      "grade": "excellent|good|fair|needs_improvement",
      "rubric_scores": {{
        "content_relevance": 0,
        "answer_structure": 0,
        "specificity_and_evidence": 0,
        "role_fit": 0,
        "communication_clarity": 0,
        "self_reflection": 0
      }},
      "strengths": ["string"],
      "weaknesses": ["string"],
      "actionable_feedback": ["string"],
      "improved_answer_example": "string"
    }}
  ],
  "final_session_feedback": {{
    "overall_summary": "string",
    "main_strengths": ["string", "string", "string"],
    "main_improvement_areas": ["string", "string", "string"],
    "practice_plan": ["string", "string", "string"],
    "final_recommendation": "string"
  }}
}}
"""


def chunk_list(items, chunk_size=6):
    return [
        items[i:i + chunk_size]
        for i in range(0, len(items), chunk_size)
    ]


def evaluate_session_batch_service(req: EvaluateSessionBatchRequest) -> Dict[str, Any]:
    if not req.answers:
        raise HTTPException(status_code=400, detail="answers must not be empty.")

    for item in req.answers:
        if not item.answer_text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"answer_text must not be empty for question_id={item.question_id}"
            )

    max_answers_per_call = 6
    answer_chunks = chunk_list(req.answers, max_answers_per_call)

    all_results = []
    genai_call_count = 0

    for chunk_index, answer_chunk in enumerate(answer_chunks, start=1):
        chunk_req = EvaluateSessionBatchRequest(
            role=req.role,
            experience_level=req.experience_level,
            language=req.language,
            answers=answer_chunk
        )

        prompt = build_batch_evaluation_prompt(chunk_req)

        try:
            data = call_gemini_json(prompt)
            genai_call_count += 1
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to evaluate session chunk {chunk_index}: {e}"
            )

        results = data.get("results", [])

        if not results:
            raise HTTPException(
                status_code=500,
                detail=f"GenAI did not return evaluation results for chunk {chunk_index}."
            )

        for item in results:
            score = int(item.get("overall_score", 0))
            score = max(0, min(100, score))

            item["overall_score"] = score
            item["grade"] = score_to_grade(score)
            item["batch_chunk_index"] = chunk_index

            all_results.append(item)

    average_score = sum(item["overall_score"] for item in all_results) / max(len(all_results), 1)
    final_session_feedback = build_rule_based_final_feedback(all_results)

    return {
        "role": req.role,
        "experience_level": req.experience_level,
        "language": req.language,
        "total_answers": len(req.answers),
        "max_answers_per_genai_call": max_answers_per_call,
        "genai_call_count_for_this_endpoint": genai_call_count,
        "average_content_score": round(float(average_score), 2),
        "results": all_results,
        "final_session_feedback": final_session_feedback
    }
