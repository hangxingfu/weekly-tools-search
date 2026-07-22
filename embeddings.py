import os
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """把一段查询文本编码为 ``VECTOR_SIZE`` 维向量。

    返回的向量必须为 ``float32`` ``ndarray``，shape ``(VECTOR_SIZE,)``。
    """

    def embed_query(self, text: str) -> np.ndarray: ...


class DashScopeEmbedding:
    """调用阿里云 DashScope 在线 embedding 接口。

    需要环境变量 ``DASHSCOPE_API_KEY``。模型名与维度由 ``config`` 注入。
    """

    def __init__(self, model: str, dimension: int, api_key: str | None = None):
        try:
            import dashscope
            from dashscope import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "缺少 dashscope SDK，请先 pip install dashscope"
            ) from error

        if api_key:
            dashscope.api_key = api_key
        elif os.environ.get("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        else:
            raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY")

        base_url = os.environ.get("DASHSCOPE_URL")
        if base_url:
            dashscope.base_http_api_url = base_url

        self._TextEmbedding = TextEmbedding
        self.model = model
        self.dimension = dimension

        print(
            f"[startup] embedding=remote model={model} dim={dimension} "
            f"base_url={dashscope.base_http_api_url}"
        )

    def embed_query(self, text: str) -> np.ndarray:
        response = self._TextEmbedding.call(
            model=self.model,
            input=text,
            dimension=self.dimension,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope 调用失败：{response.status_code} {response.message}"
            )
        embedding = response.output["embeddings"][0]["embedding"]
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


def create_embedding_backend(model: str, dimension: int) -> EmbeddingBackend:
    """工厂：当前统一走 DashScope 远程 API。"""
    return DashScopeEmbedding(model=model, dimension=dimension)
