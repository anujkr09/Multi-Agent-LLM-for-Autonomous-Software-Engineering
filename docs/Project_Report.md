# Multi-Agent LLM for Autonomous Software Engineering

## Abstract

This project presents a Machine Learning based multi-agent system that uses NLP and LLM-style reasoning to automate software engineering tasks. The system accepts a software requirement in natural language and generates requirement analysis, task planning, sample code, test cases, debugging suggestions, code review, documentation, and evaluation metrics. It is implemented in Python with Streamlit and works without a paid API key by using local ML models and deterministic fallback responses.

## Problem Statement

Manual preparation of software engineering artifacts is repetitive, time-consuming, and inconsistent. Students and developers often need requirement summaries, feature extraction, plans, boilerplate code, test cases, review notes, and documentation for each project. This project solves that problem through a structured multi-agent architecture.

## Objectives

- Understand software requirements using NLP.
- Extract features, modules, inputs, outputs, constraints, and user goals.
- Decompose requirements into development tasks.
- Generate sample Python code for the requested system.
- Generate test cases and run controlled code checks.
- Produce debugging suggestions and code review feedback.
- Generate documentation and setup instructions.
- Calculate evaluation metrics for generated output quality.

## Introduction

Large Language Models can transform natural language into useful technical artifacts. A multi-agent workflow improves traceability by assigning each software engineering activity to a specialized agent. This project demonstrates that approach through a professional Streamlit dashboard and a local ML pipeline.

## Literature Review

Research in NLP, transformer models, code generation, and agent-based systems shows that task-specialized agents can produce more interpretable and complete outputs than a single generic generation step. TF-IDF and Naive Bayes remain useful for lightweight text classification demos, while LLMs are effective for summarization, planning, and documentation.

## Proposed System

The proposed system contains seven agents: Requirement Analysis, Task Planning, Code Generation, Testing, Debugging, Code Review, and Documentation. The application supports local mock LLM responses and optional OpenAI, Hugging Face, or Gemini integration through environment variables.

## System Architecture

The architecture includes a Streamlit frontend, agent orchestrator, local ML/NLP pipeline, optional LLM provider layer, generated code execution checks, evaluation engine, dataset, and SQLite run history.

![System Architecture](../assets/system_architecture.svg)

## Methodology

1. Accept natural-language requirement from the user.
2. Apply TF-IDF, Naive Bayes classification, cosine similarity, and information extraction.
3. Route structured information through specialized agents.
4. Generate code, test cases, debugging feedback, review notes, and documentation.
5. Run syntax and smoke checks on generated code.
6. Calculate response quality and task completion metrics.
7. Display and export final output.

## Algorithms Used

- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Cosine similarity
- Rule-based information extraction
- Multi-agent task decomposition
- Syntax parsing and controlled code execution
- Weighted evaluation scoring

## Implementation Details

The project is organized into separate Python modules:

- `app.py`: Streamlit dashboard and UI pages
- `src/ml_pipeline.py`: NLP, classification, and similarity matching
- `src/agents.py`: Multi-agent orchestration
- `src/llm_provider.py`: Optional OpenAI and Hugging Face integration
- `src/code_sandbox.py`: Generated code execution checks
- `src/evaluation.py`: Quality metrics
- `src/storage.py`: SQLite run history

## Result Analysis

The system displays Accuracy, Completeness, Relevance, Code Quality, Readability, and Task Completion scores. It also provides workflow visibility for every agent and exportable JSON/Markdown output.

## Advantages

- Works without paid API keys
- Demonstrates ML, NLP, LLM, and multi-agent concepts
- Professional dashboard with multiple pages
- Includes dataset, evaluation, documentation, and run history
- Supports optional real model integration

## Limitations

- Generated code is sample-level and not production-ready by default.
- Local mock LLM output is deterministic and less creative than a real model.
- Controlled execution checks are lightweight smoke checks, not full security sandboxing.
- Dataset is suitable for demonstration, not industrial training.

## Future Scope

- Add retrieval augmented generation over existing codebases.
- Execute generated unit tests in isolated containers.
- Integrate GitHub pull request automation.
- Add advanced static analysis and vulnerability scanning.
- Deploy the app publicly on Streamlit Community Cloud.

## Conclusion

The project successfully demonstrates a professional ML-based multi-agent system for autonomous software engineering. It shows how NLP, LLM-style reasoning, code generation, testing, review, documentation, and evaluation can be combined in one complete final-year project.
