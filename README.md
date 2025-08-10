# Data Analyst Agent (evaluation-hardened)

This repository contains a FastAPI-based Data Analyst Agent API that accepts a multipart POST containing `questions.txt` and optional attachments (CSV, images, etc.) and returns JSON answers.

This version is **evaluation-hardened** to return a strict 4-element JSON array for tasks similar to the sample evaluator:
```
[ int_count, "title_string", float_corr_6dp, "data:image/png;base64,..." ]
```

## Files
- `app/main.py` — FastAPI app
- `requirements.txt` — Python dependencies
- `Dockerfile` — containerization
- `questions.txt` — example questions
- `LICENSE` — MIT

## Quick start (local)
1. Create a virtualenv (optional) and install deps:
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. Example `curl` (using the included `questions.txt`):
```bash
curl -X POST "http://localhost:8000/" \
 -F "questions=@questions.txt"
```

If testing the Wikipedia sample, ensure `questions.txt` contains the Wikipedia URL (it's already included).

## Docker
Build and run:
```bash
docker build -t data-agent .
docker run -p 8000:8000 data-agent
```

## Notes
- The API enforces a 4-element JSON array response to match the evaluation harness.
- Scatterplots use a dotted red regression line and labeled axes. Images are compressed to be under 100 KB when possible.
