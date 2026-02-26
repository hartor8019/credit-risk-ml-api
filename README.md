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

## Run with Docker
```bash
docker build -t credit-risk-ml-api .
docker run -p 8000:8000 credit-risk-ml-api