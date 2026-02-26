# Credit Risk ML API (German Credit)

End-to-end ML project: training + evaluation + deployment as a FastAPI service.

## Features
- Dataset: OpenML `credit-g`
- Model: Logistic Regression + preprocessing pipeline
- API: FastAPI `/predict` and `/health`
- Dockerized

## Run locally
```bash
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload