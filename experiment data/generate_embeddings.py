#!/usr/bin/env python3
"""Precompute normalized question embeddings for the SalesX benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit(
        'NumPy is required. Install it with: '
        'python3 -m pip install -r "experiment data/embedding_requirements.txt"'
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen3-embedding:4b"
DEFAULT_OUTPUT = ROOT / "embeddings" / "qwen3-embedding-4b"
PERIODS = ("Q1", "Q2", "Q3", "Q4", "Extra")
SOURCE_FILES = {
    "Q1": ROOT / "shared" / "Q1_qa.json",
    "Q2": ROOT / "drift_10" / "Q2_qa.json",
    "Q3": ROOT / "drift_10" / "Q3_qa.json",
    "Q4": ROOT / "drift_10" / "Q4_qa.json",
    "Extra": ROOT / "shared" / "Extra_qa.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc


def get_json(url: str, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc


def model_digest(base_url: str, model: str, timeout: float) -> str:
    tags = get_json(f"{base_url}/api/tags", timeout)
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return item.get("digest", "")
    raise RuntimeError(f"Ollama model {model!r} is not installed")


def embed_batches(
    base_url: str, model: str, texts: list[str], batch_size: int, timeout: float
) -> np.ndarray:
    batches: list[np.ndarray] = []
    expected_dimension: int | None = None
    for start in range(0, len(texts), batch_size):
        inputs = texts[start : start + batch_size]
        result = post_json(
            f"{base_url}/api/embed",
            {"model": model, "input": inputs, "truncate": False},
            timeout,
        )
        vectors = np.asarray(result.get("embeddings", []), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(inputs):
            raise RuntimeError(
                f"Expected {len(inputs)} embeddings, received shape {vectors.shape}"
            )
        if expected_dimension is None:
            expected_dimension = vectors.shape[1]
        elif vectors.shape[1] != expected_dimension:
            raise RuntimeError("Embedding dimension changed between batches")
        batches.append(vectors)
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(batches, axis=0)


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(norms)):
        raise ValueError("Embeddings contain non-finite values")
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector")
    vectors /= norms
    return vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing embedding set"
    )
    parser.add_argument(
        "--only",
        choices=PERIODS,
        help="embed one period and update an existing compatible manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    existing_outputs = list(output_dir.glob("*.npz")) if output_dir.exists() else []
    if not args.only and (manifest_path.exists() or existing_outputs) and not args.force:
        raise FileExistsError(
            f"{output_dir} already contains embeddings; pass --force to replace them"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.url.rstrip("/")
    digest = model_digest(base_url, args.model, args.timeout)
    periods = (args.only,) if args.only else PERIODS
    source_files = [SOURCE_FILES[period] for period in periods]
    missing_sources = [path.name for path in source_files if not path.exists()]
    if missing_sources:
        raise RuntimeError(
            f"Missing embedding source files: {', '.join(missing_sources)}"
        )

    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.only and manifest_path.exists()
        else None
    )
    if args.only and existing_manifest is None:
        raise RuntimeError("--only requires an existing embedding manifest")
    if existing_manifest is not None:
        if existing_manifest["model"] != args.model:
            raise RuntimeError("Existing manifest uses a different embedding model")
        if existing_manifest["ollama_model_digest"] != digest:
            raise RuntimeError("Existing manifest uses a different model digest")
        files = dict(existing_manifest["files"])
        dimension: int | None = int(existing_manifest["dimension"])
    else:
        files = {}
        dimension = None
    for source in source_files:
        records = json.loads(source.read_text(encoding="utf-8"))
        ids = [record["id"] for record in records]
        questions = [record["question"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {source.name}")
        print(f"Embedding {source.stem} ({len(records)} questions)")
        vectors = normalize(
            embed_batches(
                base_url,
                args.model,
                questions,
                args.batch_size,
                args.timeout,
            )
        )
        if dimension is None:
            dimension = vectors.shape[1]
        elif vectors.shape[1] != dimension:
            raise RuntimeError("Embedding dimension changed between quarters")

        output = output_dir / f"{source.stem.removesuffix('_qa')}.npz"
        if output.exists() and not args.force:
            raise FileExistsError(
                f"{output} already exists; pass --force to replace it"
            )
        np.savez_compressed(output, ids=np.asarray(ids), embeddings=vectors)
        files[output.name] = {
            "records": len(ids),
            "sha256": sha256(output),
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
        }

    manifest = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "ollama_model_digest": digest,
        "embedded_field": "question",
        "preprocessing": "none",
        "dtype": "float32",
        "dimension": dimension,
        "normalization": "L2",
        "similarity": "cosine (equivalent to dot product after L2 normalization)",
        "record_key": "id",
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
