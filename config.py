import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
VECTORS_PATH = BASE_DIR / "vectors.jsonl"

MODEL_NAME = "qwen3.7-text-embedding"
VECTOR_SIZE = 768
RECALL_K = 100
MIN_KEEP = 8
MIN_SCORE = 0.35
REL_RATIO = 0.80


def required_environment(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}")
    return value


def positive_environment_int(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"环境变量 {name} 必须是整数") from error
    if value <= 0:
        raise RuntimeError(f"环境变量 {name} 必须大于 0")
    return value


def load_environment(base_dir=None):
    load_dotenv((base_dir or BASE_DIR) / ".env", override=False)
    app_env = os.environ.get("APP_ENV", "development").strip().lower()
    if app_env not in {"development", "production"}:
        raise RuntimeError("APP_ENV 只能是 development 或 production")
    return {
        "app_env": app_env,
        "is_production": app_env == "production",
        "qdrant_collection": os.environ.get("QDRANT_COLLECTION", "weekly_tools"),
        "qdrant_timeout": positive_environment_int("QDRANT_TIMEOUT", 120),
    }