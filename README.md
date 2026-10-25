# VisionGuard Deepfake Detection

VisionGuard tells you whether an uploaded image is **AI-generated or real**.
It is an end-to-end MLOps project: data versioning, model training,
experiment tracking, a web app, containerization, CI and cloud deployment.

<!-- **Links:** [DagsHub project](https://dagshub.com/<user>/<RollNo>_A2_Ops) · [Live API](http://<ip>:8000/health) -->

## Stack
| Area | Tool |
|---|---|
| Model | TensorFlow/Keras: baseline CNN + fine-tuned MobileNetV2 |
| Data versioning | Git + DVC (DagsHub storage) |
| Experiment tracking | MLflow on DagsHub, model registry |
| Web app | Flask, SQLAlchemy, Postgres/SQLite history |
| Packaging | Docker, Docker Compose |
| CI | GitHub Actions (lint, tests, Docker build) |
| Deployment | Model API on cloud VM, Flask UI calls it via API key |

## Dataset
[AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/aigenerated-images-vs-real-images) (Kaggle). Tracked with DVC
