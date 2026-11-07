import os


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
    THRESHOLD = float(os.getenv("THRESHOLD", "0.31"))
    MODEL_NAME = os.getenv("MODEL_NAME", "visionguard-mobilenetv2-v2 (lr 1e-4)")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///history.db")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024          # 5 MB upload limit
    ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
    INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")
    API_URL = os.getenv("API_URL", "")
    API_KEY = os.getenv("API_KEY", "")
