from __future__ import annotations

import pandas as pd

from src.agents import MultiAgentSoftwareEngineer
from src.code_sandbox import run_generated_code_checks
from src.evaluation import evaluate_pipeline
from src.llm_provider import LLMProvider, normalize_provider
from src.ml_pipeline import RequirementNLP


def test_dataset_has_enough_demo_rows() -> None:
    dataset = pd.read_csv("data/sample_requirements.csv")
    assert len(dataset) >= 25
    assert dataset["domain"].nunique() >= 10


def test_requirement_nlp_classifies_and_matches() -> None:
    nlp = RequirementNLP()
    requirement = "Build a hospital appointment booking system with doctor slots and patient confirmation."
    domain = nlp.classify_domain(requirement)
    match = nlp.find_closest_requirement(requirement)
    assert domain
    assert match.similarity > 20
    assert match.requirement_id


def test_local_provider_fallback_never_requires_key() -> None:
    provider = LLMProvider("Local Mock LLM")
    response = provider.generate("prompt", "fallback text")
    assert response.text == "fallback text"
    assert response.used_real_model is False


def test_provider_aliases_include_gemini() -> None:
    assert normalize_provider("gemini") == "Gemini"
    assert normalize_provider("google") == "Gemini"
    assert LLMProvider("gemini").provider == "Gemini"


def test_multi_agent_pipeline_returns_complete_result() -> None:
    engine = MultiAgentSoftwareEngineer()
    result = engine.run("Create an online quiz system with objective questions, scoring, and result report.")
    assert result["analysis"]["features"]
    assert len(result["plan"]["tasks"]) >= 6
    assert "class " in result["code"]
    assert len(result["tests"]) >= 5
    assert result["execution"]["overall_status"] == "Passed"
    assert result["documentation"]["overview"]


def test_evaluation_scores_are_bounded() -> None:
    engine = MultiAgentSoftwareEngineer()
    result = engine.run("Build a courier tracking dashboard with parcel status and delivery updates.")
    metrics = evaluate_pipeline(result)
    assert set(metrics) == {
        "Accuracy Score",
        "Completeness Score",
        "Relevance Score",
        "Code Quality Score",
        "Readability Score",
        "Task Completion Score",
    }
    assert all(0 <= score <= 100 for score in metrics.values())


def test_generated_code_sandbox_rejects_invalid_code() -> None:
    report = run_generated_code_checks("def broken(:\n    pass")
    assert report["overall_status"] == "Failed"
