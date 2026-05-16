from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerateQuestionsRequest(BaseModel):
    role: str = Field(..., json_schema_extra={"example": "Machine Learning Engineer"})
    experience_level: str = Field("fresh graduate", json_schema_extra={"example": "fresh graduate"})
    language: str = Field("en", json_schema_extra={"example": "en"})
    num_questions_per_type: int = Field(1, ge=1, le=3)
    question_types: Optional[List[str]] = None
    user_profile_summary: Optional[str] = None


class InterviewQuestion(BaseModel):
    question_id: str
    interview_type: str
    question: str
    purpose: str
    expected_answer_points: List[str]


class GenerateQuestionsResponse(BaseModel):
    session_id: str
    role: str
    experience_level: str
    language: str
    questions: List[InterviewQuestion]
    rubric: Dict[str, Any]


class BatchAnswerItem(BaseModel):
    question_id: str
    interview_type: str
    question: str
    answer_text: str


class EvaluateSessionBatchRequest(BaseModel):
    role: str = Field(..., json_schema_extra={"example": "Machine Learning Engineer"})
    experience_level: str = Field("fresh graduate", json_schema_extra={"example": "fresh graduate"})
    language: str = Field("en", json_schema_extra={"example": "en"})
    answers: List[BatchAnswerItem]
