import json
import os

import numpy as np

from config import RECALL_K, VECTOR_SIZE, required_environment, positive_environment_int
from ranking import build_site_meta


class LocalSearchBackend:
    """开发环境：把 ``vectors.jsonl`` 一次性加载到内存做矩阵点积。"""

    def __init__(self, vectors_path):
        print(f"[startup] loading vectors from {vectors_path}")
        with vectors_path.open("r", encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file]
        self.matrix = np.array(
            [record["vector"] for record in self.records], dtype=np.float32
        )
        self.site_meta = build_site_meta(self.records)
        print(
            f"[startup] vectors loaded: {len(self.records)} records, "
            f"dim={self.matrix.shape[1]}"
        )

    def search(self, query_vector):
        similarities = self.matrix @ query_vector
        order = np.argsort(-similarities)[:RECALL_K]
        return [
            (float(similarities[index]), self.records[index]) for index in order
        ]

    def close(self):
        pass


class QdrantSearchBackend:
    """生产环境：远程 Qdrant 集合，余弦距离需与 ``VECTOR_SIZE`` 一致。"""

    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance
        except ImportError as error:
            raise RuntimeError(
                "生产环境缺少 qdrant-client，请先安装该依赖"
            ) from error

        self.collection_name = os.environ.get("QDRANT_COLLECTION", "weekly_tools")
        timeout = positive_environment_int("QDRANT_TIMEOUT", 120)
        self.client = QdrantClient(
            url=required_environment("QDRANT_URL"),
            api_key=required_environment("QDRANT_API_KEY"),
            timeout=timeout,
        )

        collection = self.client.get_collection(self.collection_name)
        vectors_config = collection.config.params.vectors
        if isinstance(vectors_config, dict):
            raise RuntimeError(
                f"集合 {self.collection_name} 使用命名向量，与当前检索配置不兼容"
            )
        if (
            vectors_config.size != VECTOR_SIZE
            or vectors_config.distance != Distance.COSINE
        ):
            raise RuntimeError(
                f"集合 {self.collection_name} 配置不兼容：期望 "
                f"{VECTOR_SIZE} 维 COSINE，实际为 "
                f"{vectors_config.size} 维 {vectors_config.distance}"
            )

        self.site_meta = self._load_site_meta()
        print(
            f"[startup] qdrant ready: collection={self.collection_name}, "
            f"points={self.site_meta['total_records']}"
        )

    def _load_site_meta(self):
        total_records = self.client.count(
            collection_name=self.collection_name, exact=True
        ).count
        records = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=["source_file", "date"],
                with_vectors=False,
            )
            records.extend(point.payload or {} for point in points)
            if offset is None:
                break

        return build_site_meta(records, total_records=total_records)

    def search(self, query_vector):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            with_payload=True,
            with_vectors=False,
            limit=RECALL_K,
        )
        return [
            (float(point.score), point.payload or {}) for point in response.points
        ]

    def close(self):
        self.client.close()