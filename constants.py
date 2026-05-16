EVALIFY_RUBRIC = {
    "content_relevance": {"weight": 25, "description": "How relevant the answer is to the question and selected role."},
    "answer_structure": {"weight": 20, "description": "How clearly the answer is structured, including opening, main points, examples, and conclusion. For behavioral questions, consider the STAR method."},
    "specificity_and_evidence": {"weight": 20, "description": "How specific the answer is, including concrete examples, projects, measurable impact, or evidence."},
    "role_fit": {"weight": 15, "description": "How well the answer demonstrates fit with the selected role's responsibilities and expected competencies."},
    "communication_clarity": {"weight": 10, "description": "How clear, focused, and easy to understand the answer is."},
    "self_reflection": {"weight": 10, "description": "How well the answer shows reflection, learning, initiative, and improvement mindset."},
}

QUESTION_TYPES = {
    "hr": "General HR questions about motivation, background, personality, and career goals.",
    "behavioral": "Past-experience questions, usually best answered using the STAR method.",
    "situational": "Scenario-based questions to assess decision-making and problem-solving.",
    "technical": "Technical questions based on the selected role.",
    "role_specific": "Questions specific to the responsibilities and core skills of the selected role.",
    "communication": "Questions to assess how clearly the candidate explains ideas.",
}
