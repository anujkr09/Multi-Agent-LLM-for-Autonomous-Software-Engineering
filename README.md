# Multi-Agent LLM for Autonomous Software Engineering

A professional Streamlit-based Machine Learning project where multiple LLM-style agents collaborate to automate requirement analysis, task planning, code generation, testing, debugging, code review, documentation, and evaluation.

The project is designed for final-year ML project submission and presentation. It works without a paid API key by using local NLP models, scikit-learn evaluation logic, and deterministic mock LLM responses.

## Features

- Requirement Analysis Agent with NLP extraction
- Task Planning Agent for task decomposition
- Code Generation Agent for sample Python code
- Testing Agent for generated test scenarios
- Debugging Agent for issue detection and fixes
- Code Review Agent for quality, security, and performance feedback
- Documentation Agent for generated project documentation
- Evaluation dashboard with charts and progress bars
- SQLite history of recent runs
- Sample dataset for ML demonstration

## Tech Stack

- Frontend: Streamlit
- Backend: Python
- ML/NLP: scikit-learn TF-IDF, Naive Bayes, cosine similarity
- Storage: SQLite and CSV dataset
- Visualization: Plotly

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The demo dataset is available at `data/sample_requirements.csv` and contains software requirements with expected outputs and domains.

## Note

This project intentionally avoids mandatory paid APIs. A real LLM provider can be connected later, but the submitted project remains runnable on any normal Python environment.
