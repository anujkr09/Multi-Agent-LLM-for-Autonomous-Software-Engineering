from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATASET_PATH = Path("data/sample_requirements.csv")


@dataclass(frozen=True)
class DatasetMatch:
    requirement_id: str
    domain: str
    similarity: float
    expected_code_type: str
    expected_test_cases: str


class RequirementNLP:
    """Small local NLP layer used when no external LLM key is available."""

    def __init__(self, dataset_path: Path = DATASET_PATH) -> None:
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.dataset["user_requirement"])
        self.domain_model: Pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
                ("clf", MultinomialNB()),
            ]
        )
        self.domain_model.fit(self.dataset["user_requirement"], self.dataset["domain"])

    def classify_domain(self, requirement: str) -> str:
        return str(self.domain_model.predict([requirement])[0])

    def find_closest_requirement(self, requirement: str) -> DatasetMatch:
        query = self.vectorizer.transform([requirement])
        scores = cosine_similarity(query, self.matrix)[0]
        best_index = int(scores.argmax())
        row = self.dataset.iloc[best_index]
        return DatasetMatch(
            requirement_id=str(row["requirement_id"]),
            domain=str(row["domain"]),
            similarity=round(float(scores[best_index]) * 100, 2),
            expected_code_type=str(row["expected_code_type"]),
            expected_test_cases=str(row["expected_test_cases"]),
        )

    def extract_keywords(self, requirement: str, top_n: int = 10) -> list[str]:
        query = self.vectorizer.transform([requirement])
        names = self.vectorizer.get_feature_names_out()
        weights = query.toarray()[0]
        ranked = weights.argsort()[::-1]
        keywords = [names[index] for index in ranked if weights[index] > 0]
        if len(keywords) < 4:
            keywords.extend(self._fallback_keywords(requirement))
        return list(dict.fromkeys(keywords[:top_n]))

    @staticmethod
    def extract_entities(requirement: str) -> dict[str, list[str]]:
        lowered = requirement.lower()
        entity_map = {
            "users": ["admin", "student", "teacher", "faculty", "patient", "doctor", "member", "customer", "manager"],
            "actions": ["create", "build", "track", "book", "issue", "return", "login", "generate", "manage", "update", "delete"],
            "outputs": ["report", "dashboard", "summary", "chart", "notification", "export", "confirmation"],
            "constraints": ["secure", "monthly", "role-based", "real-time", "responsive", "category-wise", "minimal"],
        }
        return {
            group: [word for word in words if re.search(rf"\b{re.escape(word)}\b", lowered)]
            for group, words in entity_map.items()
        }

    @staticmethod
    def _fallback_keywords(text: str) -> Iterable[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
        stop_words = {"with", "and", "for", "the", "that", "this", "from", "using", "into", "system"}
        return [token for token in tokens if token not in stop_words]
