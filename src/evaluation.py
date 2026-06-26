from __future__ import annotations

import ast
from statistics import mean
from typing import Any


def evaluate_pipeline(result: dict[str, Any]) -> dict[str, float]:
    analysis = result["analysis"]
    code = result["code"]
    tests = result["tests"]
    execution_passed = result.get("execution", {}).get("overall_status") == "Passed"

    syntax_score = _syntax_score(code)
    completeness = _bounded(len(analysis.get("features", [])) * 12 + len(result["plan"].get("tasks", [])) * 4)
    relevance = _bounded(analysis["dataset_match"]["similarity"] + 35)
    code_quality = _bounded(syntax_score * 0.45 + _structure_score(code) * 0.4 + (15 if execution_passed else 0))
    readability = _bounded(70 + min(25, len(code.splitlines()) / 2))
    task_completion = _bounded(len(tests) * 12 + len(result["documentation"]) * 8)
    accuracy = _bounded(mean([syntax_score, completeness, relevance, code_quality, readability, task_completion]))

    return {
        "Accuracy Score": round(accuracy, 2),
        "Completeness Score": round(completeness, 2),
        "Relevance Score": round(relevance, 2),
        "Code Quality Score": round(code_quality, 2),
        "Readability Score": round(readability, 2),
        "Task Completion Score": round(task_completion, 2),
    }


def _syntax_score(code: str) -> float:
    try:
        ast.parse(code)
    except SyntaxError:
        return 35.0
    return 95.0


def _structure_score(code: str) -> float:
    score = 45
    for token in ["class ", "def ", "dataclass", "raise ValueError", "List[", "Dict["]:
        if token in code:
            score += 8
    return _bounded(score)


def _bounded(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))
