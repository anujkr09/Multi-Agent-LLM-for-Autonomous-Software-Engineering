# Deployment Guide

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional Environment Variables

For OpenAI:

```powershell
setx OPENAI_API_KEY "your_api_key"
setx OPENAI_MODEL "gpt-5.5"
```

For Hugging Face:

```powershell
setx HF_API_TOKEN "your_token"
setx HF_MODEL "google/flan-t5-base"
```

For Gemini:

```powershell
setx AI_PROVIDER "gemini"
setx GEMINI_API_KEY "your_api_key"
setx GEMINI_MODEL "gemini-2.5-flash"
```

If these keys are not configured, the app uses the Local Mock LLM provider and still works.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set main file path to `app.py`.
5. Add optional secrets for `OPENAI_API_KEY`, `HF_API_TOKEN`, or `GEMINI_API_KEY`.
6. Deploy.

## Health Check

After deployment, open the app and test:

- Home page loads
- Requirement Input runs the workflow
- Output page shows generated artifacts
- Evaluation page displays charts
- About Project page shows report and dataset
