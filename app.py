import atexit

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import config
from backends import LocalSearchBackend, QdrantSearchBackend
from embeddings import create_embedding_backend
from ranking import select_results
import logging

ENV = config.load_environment()
APP_ENV = ENV["app_env"]
IS_PRODUCTION = ENV["is_production"]
SEARCH_BACKEND = (
    QdrantSearchBackend() if IS_PRODUCTION else LocalSearchBackend(config.VECTORS_PATH)
)
SITE_META = SEARCH_BACKEND.site_meta
atexit.register(SEARCH_BACKEND.close)

EMBEDDER = create_embedding_backend(
    model=config.MODEL_NAME,
    dimension=config.VECTOR_SIZE,
)
print(f"[startup] model ready: environment={APP_ENV}")

app = Flask(__name__, static_folder=None)


@app.route("/api/meta")
def meta():
    return jsonify(SITE_META)


@app.route("/api/search")
def search():
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "missing q"}), 400

    try:
        query_vector = EMBEDDER.embed_query(query)
        if len(query_vector) != config.VECTOR_SIZE:
            raise RuntimeError(
                f"查询向量维度错误：期望 {config.VECTOR_SIZE}，实际为 {len(query_vector)}"
            )
        candidates = SEARCH_BACKEND.search(query_vector)
        items, scores, cutoff, fallback = select_results(candidates)
    except RuntimeError as error:
        app.logger.error("search config error: %s", error)
        return jsonify({"error": "search service unavailable"}), 502
    except Exception:
        app.logger.exception("检索服务调用失败")
        return jsonify({"error": "search service unavailable"}), 502

    print(
        f"search environment={APP_ENV} cutoff={cutoff} fallback={fallback} "
        f"top10={[round(s,4) for s in scores[:10]]} "
        f"top30={[round(s,4) for s in scores[29:31]]} "  # 看第29、30的分数
        f"last={[round(s,4) for s in scores[-3:]]}",  # 看尾部
        flush=True,
    )
    return jsonify(items)


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(config.DIST_DIR / "assets", filename)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(config.DIST_DIR, "favicon.ico")


@app.route("/")
def index():
    return send_from_directory(config.DIST_DIR, "index.html")


@app.route("/<path:path>")
def spa_fallback(path):
    full_path = config.DIST_DIR / path
    if full_path.is_file():
        return send_from_directory(config.DIST_DIR, path)
    return send_from_directory(config.DIST_DIR, "index.html")


app.logger.setLevel(logging.INFO)
