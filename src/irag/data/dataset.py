from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from irag.core.models import Quarter, QuarterBatch, TicketRecord


class SalesXDataset:
    def __init__(self, data_dir: Path, embedding_model: str) -> None:
        self.data_dir = data_dir
        self.embedding_dir = data_dir / "embeddings" / embedding_model.replace(":", "-")
        self._canonical_records: dict[str, TicketRecord] | None = None
        self._vectors: dict[str, np.ndarray] | None = None
        self._checksums_verified = False

    def load_all_quarters(self, include_extra: bool = True) -> list[QuarterBatch]:
        periods = [
            period
            for period in Quarter
            if include_extra or period != Quarter.EXTRA
        ]
        return [self.load_quarter(period) for period in periods]

    def load_quarter(self, quarter: Quarter) -> QuarterBatch:
        path = self.data_dir / f"{quarter.value}_qa.json"
        records = [TicketRecord.model_validate(item) for item in self._load_json(path)]
        return QuarterBatch(quarter=quarter, records=records)

    def vector(self, record_id: str) -> np.ndarray:
        if self._vectors is None:
            self._vectors = self._load_vectors()
        try:
            return self._vectors[record_id]
        except KeyError as exc:
            raise ValueError(
                f"No precomputed embedding exists for {record_id}"
            ) from exc

    def validate_batches(self, batches: list[QuarterBatch]) -> None:
        self.verify_checksums()
        canonical = self._get_canonical_records()
        for batch in batches:
            for record in batch.records:
                expected = canonical.get(record.id)
                if expected is None:
                    raise ValueError(
                        f"Record {record.id} is not part of the SalesX benchmark"
                    )
                if record.question != expected.question:
                    raise ValueError(
                        f"Question text for {record.id} differs from the text used to create its embedding"
                    )
                self.vector(record.id)

    def manifest(self) -> dict:
        return self._load_json(self.data_dir / "manifest.json")

    def embedding_manifest(self) -> dict:
        return self._load_json(self.embedding_dir / "manifest.json")

    def verify_checksums(self) -> None:
        if self._checksums_verified:
            return
        manifest = self.embedding_manifest()
        for filename, details in manifest["files"].items():
            path = self.embedding_dir / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != details["sha256"]:
                raise ValueError(f"Embedding checksum mismatch for {filename}")
            source_path = self.data_dir / details["source"]
            source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if source_digest != details["source_sha256"]:
                raise ValueError(f"Embedding source checksum mismatch for {filename}")
        self._checksums_verified = True

    def _get_canonical_records(self) -> dict[str, TicketRecord]:
        if self._canonical_records is None:
            self._canonical_records = {
                record.id: record
                for batch in self.load_all_quarters()
                for record in batch.records
            }
        return self._canonical_records

    def _load_vectors(self) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for period in Quarter:
            path = self.embedding_dir / f"{period.value}.npz"
            with np.load(path, allow_pickle=False) as archive:
                ids = archive["ids"]
                embeddings = archive["embeddings"].astype(np.float32, copy=False)
            if len(ids) != len(embeddings):
                raise ValueError(f"IDs and embeddings are misaligned in {path.name}")
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.all(np.isfinite(norms)) or np.any(norms == 0):
                raise ValueError(f"Invalid embedding vector in {path.name}")
            embeddings = embeddings / norms[:, np.newaxis]
            for record_id, vector in zip(ids.tolist(), embeddings, strict=True):
                record_key = str(record_id)
                if record_key in vectors:
                    raise ValueError(f"Duplicate embedding ID {record_key}")
                vectors[record_key] = vector
        return vectors

    @staticmethod
    def _load_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Required benchmark file is missing: {path}") from exc
