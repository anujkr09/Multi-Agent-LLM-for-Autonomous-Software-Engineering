from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.ml_pipeline import RequirementNLP


@dataclass
class AgentResult:
    name: str
    role: str
    status: str
    output: dict[str, Any] = field(default_factory=dict)


class MultiAgentSoftwareEngineer:
    def __init__(self) -> None:
        self.nlp = RequirementNLP()

    def run(self, requirement: str) -> dict[str, Any]:
        analysis = self.requirement_analysis(requirement)
        plan = self.task_planning(requirement, analysis)
        code = self.code_generation(requirement, analysis, plan)
        tests = self.testing(requirement, analysis, code)
        debugging = self.debugging(code)
        review = self.code_review(code)
        docs = self.documentation(requirement, analysis, plan, code, tests)
        return {
            "requirement": requirement,
            "analysis": analysis,
            "plan": plan,
            "code": code,
            "tests": tests,
            "debugging": debugging,
            "review": review,
            "documentation": docs,
        }

    def workflow_results(self, result: dict[str, Any]) -> list[AgentResult]:
        return [
            AgentResult("Requirement Analysis Agent", "NLP extraction and domain classification", "Completed", result["analysis"]),
            AgentResult("Task Planning Agent", "Task decomposition and sprint planning", "Completed", result["plan"]),
            AgentResult("Code Generation Agent", "Template-based local LLM style generation", "Completed", {"language": "Python", "lines": len(result["code"].splitlines())}),
            AgentResult("Testing Agent", "Functional and validation test generation", "Completed", {"test_count": len(result["tests"])}),
            AgentResult("Debugging Agent", "Static issue detection and fix suggestions", "Completed", result["debugging"]),
            AgentResult("Code Review Agent", "Quality, security, readability review", "Completed", result["review"]),
            AgentResult("Documentation Agent", "README and module explanation generation", "Completed", {"sections": list(result["documentation"].keys())}),
        ]

    def requirement_analysis(self, requirement: str) -> dict[str, Any]:
        keywords = self.nlp.extract_keywords(requirement)
        entities = self.nlp.extract_entities(requirement)
        domain = self.nlp.classify_domain(requirement)
        match = self.nlp.find_closest_requirement(requirement)
        features = self._derive_features(requirement, keywords, entities)
        modules = [f"{feature.title()} Module" for feature in features[:5]]
        return {
            "summary": self._summarize_requirement(requirement),
            "domain": domain,
            "dataset_match": match.__dict__,
            "keywords": keywords,
            "entities": entities,
            "features": features,
            "modules": modules,
            "inputs": self._derive_inputs(entities, features),
            "outputs": self._derive_outputs(entities),
            "constraints": entities.get("constraints") or ["offline demo compatible", "responsive user interface", "basic validation"],
            "user_goals": ["reduce manual effort", "organize records", "generate useful software outputs"],
        }

    def task_planning(self, requirement: str, analysis: dict[str, Any]) -> dict[str, Any]:
        modules = analysis["modules"] or ["Core Module"]
        tasks = [
            {"phase": "Requirement Understanding", "task": "Extract domain, users, features, inputs, outputs, and constraints."},
            {"phase": "Architecture", "task": "Design Streamlit UI, local NLP layer, SQLite/JSON storage, and agent orchestration."},
        ]
        tasks.extend({"phase": "Implementation", "task": f"Build {module} with validation and clean output views."} for module in modules)
        tasks.extend(
            [
                {"phase": "Testing", "task": "Create positive, negative, boundary, and integration test cases."},
                {"phase": "Review", "task": "Check code quality, security, readability, and maintainability."},
                {"phase": "Documentation", "task": "Generate setup guide, module explanation, and future scope."},
            ]
        )
        return {
            "methodology": "NLP preprocessing -> multi-agent reasoning -> generated artifacts -> metric evaluation",
            "tasks": tasks,
            "estimated_modules": modules,
            "architecture": ["Streamlit Frontend", "Agent Orchestrator", "NLP/ML Pipeline", "Evaluation Engine", "SQLite Storage"],
        }

    def code_generation(self, requirement: str, analysis: dict[str, Any], plan: dict[str, Any]) -> str:
        features = analysis["features"][:4] or ["records", "reports"]
        model_name = self._class_name(requirement)
        fields = ["id", *[self._field_name(feature) for feature in features]]
        field_lines = "\n".join(f"    {field}: str" for field in fields)
        return f'''from dataclasses import dataclass
from typing import Dict, List


@dataclass
class {model_name}:
{field_lines}


class {model_name}Service:
    def __init__(self) -> None:
        self.records: Dict[str, {model_name}] = {{}}

    def create(self, record: {model_name}) -> {model_name}:
        if not record.id:
            raise ValueError("Record id is required")
        self.records[record.id] = record
        return record

    def list_all(self) -> List[{model_name}]:
        return list(self.records.values())

    def search(self, keyword: str) -> List[{model_name}]:
        keyword = keyword.lower().strip()
        return [
            record for record in self.records.values()
            if keyword in str(record).lower()
        ]

    def generate_report(self) -> dict:
        return {{
            "total_records": len(self.records),
            "features": {features!r},
            "domain": "{analysis["domain"]}",
        }}
'''

    def testing(self, requirement: str, analysis: dict[str, Any], code: str) -> list[dict[str, str]]:
        feature = (analysis["features"] or ["record"])[0]
        return [
            {"type": "Positive", "case": f"Create a valid {feature} record and verify it is saved.", "expected": "Record is stored and visible in list view."},
            {"type": "Negative", "case": "Submit the form without required id.", "expected": "Validation error is shown."},
            {"type": "Boundary", "case": "Search using blank and mixed-case keywords.", "expected": "Blank search is handled and mixed-case search works."},
            {"type": "Integration", "case": "Create multiple records and generate report.", "expected": "Report count matches stored records."},
            {"type": "Security", "case": "Enter script-like text in input fields.", "expected": "Text is treated as plain data, not executable code."},
        ]

    @staticmethod
    def debugging(code: str) -> dict[str, Any]:
        issues = []
        if "raise ValueError" not in code:
            issues.append("Missing required-field validation.")
        if "Dict[str" not in code:
            issues.append("Storage type is not explicit.")
        return {
            "issues_found": issues or ["No critical syntax-level issues detected in generated sample."],
            "suggested_fixes": [
                "Add persistent SQLite storage for production use.",
                "Add role-based authentication before exposing admin operations.",
                "Write pytest unit tests for service methods.",
            ],
        }

    @staticmethod
    def code_review(code: str) -> dict[str, Any]:
        line_count = len(code.splitlines())
        return {
            "quality": "Good" if line_count < 80 else "Moderate",
            "readability": "Clear class-based structure with typed methods.",
            "security": "Basic input validation is present; authentication and authorization should be added for deployment.",
            "performance": "In-memory search is acceptable for demo data; use indexed database queries at scale.",
            "maintainability": "Service layer can be tested independently from the UI.",
        }

    def documentation(self, requirement: str, analysis: dict[str, Any], plan: dict[str, Any], code: str, tests: list[dict[str, str]]) -> dict[str, str]:
        feature_text = ", ".join(analysis["features"])
        module_text = ", ".join(plan["estimated_modules"])
        return {
            "overview": f"This project automates software engineering artifacts for: {requirement}",
            "setup": "Install dependencies with `pip install -r requirements.txt` and run `streamlit run app.py`.",
            "modules": f"Main generated modules include {module_text}. Extracted features include {feature_text}.",
            "testing": f"The Testing Agent generated {len(tests)} functional, boundary, integration, and security-oriented test cases.",
            "deployment_notes": "For a production version, connect persistent storage, authentication, CI tests, and a real LLM provider.",
        }

    @staticmethod
    def _summarize_requirement(requirement: str) -> str:
        cleaned = re.sub(r"\s+", " ", requirement).strip()
        return cleaned[:180] + ("..." if len(cleaned) > 180 else "")

    @staticmethod
    def _derive_features(requirement: str, keywords: list[str], entities: dict[str, list[str]]) -> list[str]:
        phrases = []
        lowered = requirement.lower()
        known_features = {
            "login": "authentication",
            "report": "reporting",
            "dashboard": "dashboard analytics",
            "book": "booking or catalog management",
            "appointment": "appointment scheduling",
            "expense": "expense tracking",
            "attendance": "attendance tracking",
            "product": "product management",
            "inventory": "inventory tracking",
            "return": "return workflow",
            "issue": "issue workflow",
        }
        for word, feature in known_features.items():
            if word in lowered:
                phrases.append(feature)
        phrases.extend(keyword for keyword in keywords if len(keyword.split()) <= 2)
        return list(dict.fromkeys(phrases))[:8] or ["record management", "search", "reporting"]

    @staticmethod
    def _derive_inputs(entities: dict[str, list[str]], features: list[str]) -> list[str]:
        users = entities.get("users") or ["user"]
        return [f"{user} details" for user in users[:3]] + [f"{feature} data" for feature in features[:3]]

    @staticmethod
    def _derive_outputs(entities: dict[str, list[str]]) -> list[str]:
        outputs = entities.get("outputs") or ["dashboard", "report", "summary"]
        return [output.title() for output in outputs]

    @staticmethod
    def _class_name(requirement: str) -> str:
        words = re.findall(r"[A-Za-z]+", requirement.title())
        useful = [word for word in words if word.lower() not in {"create", "build", "with", "and", "for", "the", "a", "an"}]
        return "".join(useful[:3]) or "GeneratedSystem"

    @staticmethod
    def _field_name(feature: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", feature.lower()).strip("_") or "value"
