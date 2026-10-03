from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
from qdrant_client import QdrantClient, models

from irag.engine.temporal_pruning import SemanticCandidate


@dataclass(frozen=True)
class CollectionStatus:
    status: str
    points_count: int
    indexed_vectors_count: int


class QdrantVectorStore:
    """Narrow Qdrant adapter used by the temporal-retrieval benchmark."""

    def __init__(
        self,
        *,
        url: str,
        grpc_port: int,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.url = url.rstrip("/")
        self._client = QdrantClient(
            url=self.url,
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=timeout_seconds,
        )
        self._http = httpx.Client(base_url=self.url, timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()
        self._http.close()

    def ready(self) -> bool:
        try:
            response = self._http.get("/readyz")
            return response.is_success
        except httpx.HTTPError:
            return False

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 30.0,
        poll_seconds: float = 0.5,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.ready():
                return
            time.sleep(poll_seconds)
        raise TimeoutError(f"Qdrant is not ready at {self.url}")

    def recreate_collection(
        self,
        *,
        collection_name: str,
        dimension: int,
        hnsw_m: int,
        ef_construct: int,
        full_scan_threshold: int,
        indexing_threshold: int,
    ) -> None:
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=hnsw_m,
                ef_construct=ef_construct,
                full_scan_threshold=full_scan_threshold,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )
        self._client.create_payload_index(
            collection_name=collection_name,
            field_name="insertion_index",
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True,
        )

    def upsert_batch(
        self,
        *,
        collection_name: str,
        point_ids: list[int],
        insertion_indices: list[int],
        vectors: np.ndarray,
    ) -> None:
        payloads = [
            {"insertion_index": insertion_index}
            for insertion_index in insertion_indices
        ]
        self._client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=point_ids,
                vectors=vectors.tolist(),
                payloads=payloads,
            ),
            wait=True,
        )

    def query(
        self,
        *,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int,
        semantic_threshold: float,
        hnsw_ef: int,
        minimum_insertion_index: int | None = None,
        exact: bool = False,
    ) -> list[SemanticCandidate]:
        query_filter = None
        if minimum_insertion_index is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="insertion_index",
                        range=models.Range(gte=minimum_insertion_index),
                    )
                ]
            )
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            search_params=models.SearchParams(hnsw_ef=hnsw_ef, exact=exact),
            limit=limit,
            score_threshold=semantic_threshold,
            with_payload=["insertion_index"],
            with_vectors=False,
        )
        return [
            SemanticCandidate(
                point_id=int(point.id),
                insertion_index=int(point.payload["insertion_index"]),
                cosine_similarity=float(point.score),
            )
            for point in response.points
        ]

    def collection_status(self, collection_name: str) -> CollectionStatus:
        info = self._client.get_collection(collection_name)
        status = getattr(info.status, "value", str(info.status))
        return CollectionStatus(
            status=str(status),
            points_count=int(info.points_count or 0),
            indexed_vectors_count=int(info.indexed_vectors_count or 0),
        )

    def wait_until_indexed(
        self,
        *,
        collection_name: str,
        expected_points: int,
        timeout_seconds: float,
        poll_seconds: float = 1.0,
    ) -> CollectionStatus:
        deadline = time.monotonic() + timeout_seconds
        last_status = self.collection_status(collection_name)
        while time.monotonic() < deadline:
            last_status = self.collection_status(collection_name)
            indexed_enough = (
                last_status.indexed_vectors_count >= int(expected_points * 0.95)
            )
            if (
                last_status.status.lower() == "green"
                and last_status.points_count == expected_points
                and indexed_enough
            ):
                return last_status
            time.sleep(poll_seconds)
        raise TimeoutError(
            "Qdrant did not finish indexing before the timeout: "
            f"{last_status!r}"
        )

    def collection_memory(self, collection_name: str) -> dict[str, Any] | None:
        response = self._http.get(f"/collections/{collection_name}/memory")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def resident_memory_bytes(self) -> int | None:
        response = self._http.get("/metrics")
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.startswith("memory_resident_bytes "):
                return int(float(line.split(maxsplit=1)[1]))
        return None

    def version(self) -> str | None:
        response = self._http.get("/")
        response.raise_for_status()
        body = response.json()
        return body.get("version")
