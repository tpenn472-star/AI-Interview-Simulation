from typing import Any, Dict, List


def score_to_grade(score: int) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "fair"
    return "needs_improvement"


def combine_content_delivery_score(content_score: float, delivery_score: float) -> Dict[str, Any]:
    final_score = round((0.7 * content_score) + (0.3 * delivery_score), 2)

    return {
        "content_score": content_score,
        "delivery_score": delivery_score,
        "final_score": final_score,
        "formula": "final_score = 70% content_score + 30% delivery_score"
    }


def build_rule_based_final_feedback(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "overall_summary": "No evaluation results were available.",
            "main_strengths": [],
            "main_improvement_areas": [],
            "practice_plan": [],
            "final_recommendation": "Complete the interview answers before requesting feedback."
        }

    scores = [int(item.get("overall_score", 0)) for item in results]
    avg_score = sum(scores) / max(len(scores), 1)

    all_strengths = []
    all_weaknesses = []
    all_actions = []

    for item in results:
        all_strengths.extend(item.get("strengths", []))
        all_weaknesses.extend(item.get("weaknesses", []))
        all_actions.extend(item.get("actionable_feedback", []))

    main_strengths = all_strengths[:3] if all_strengths else [
        "The candidate provided relevant answers for the selected role."
    ]

    main_improvement_areas = all_weaknesses[:3] if all_weaknesses else [
        "Add more specific examples, measurable outcomes, and clearer reasoning where possible."
    ]

    practice_plan = all_actions[:3] if all_actions else [
        "Practice answering with a clear structure: context, action, result, and learning.",
        "Prepare specific project examples that demonstrate role-related skills.",
        "Add measurable outcomes or concrete impact when explaining past work."
    ]

    if avg_score >= 85:
        recommendation = "Strong performance. The candidate is well-prepared for this interview type."
    elif avg_score >= 70:
        recommendation = "Good performance. The candidate is mostly ready but should improve answer depth and specificity."
    elif avg_score >= 55:
        recommendation = "Fair performance. The candidate should practice structure, specificity, and role alignment."
    else:
        recommendation = "Needs improvement. The candidate should prepare stronger examples and more complete answers."

    return {
        "overall_summary": f"The candidate completed {len(results)} answers with an average content score of {avg_score:.2f}.",
        "main_strengths": main_strengths,
        "main_improvement_areas": main_improvement_areas,
        "practice_plan": practice_plan,
        "final_recommendation": recommendation
    }
