from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from irag.client.qdrant import QdrantVectorStore
from irag.engine.temporal_pruning import (
    WeightedCandidate,
    certified_newest_first_top_k,
    exhaustive_weighted_top_k,
    recall_at_k,
    rerank_candidates,
    temporal_pruning_is_safe,
)

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_DIR = (
    REPOSITORY_ROOT / "experiment data" / "embeddings" / "qwen3-embedding-4b"
)


@dataclass(frozen=True)
class BenchmarkConfig:
    qdrant_url: str
    qdrant_grpc_port: int
    collection_name: str
    embedding_dir: Path
    output_dir: Path
    points: int
    queries: int
    seed: int
    noise_std: float
    cluster_size: int
    top_k: int
    semantic_threshold: float
    lambdas: tuple[float, ...]
    candidate_limit: int
    certified_scan_block: int
    fixed_window: int
    initial_window: int
    window_growth: float
    hnsw_m: int
    ef_construct: int
    hnsw_ef: int
    full_scan_threshold: int
    indexing_threshold: int
    batch_size: int
    concurrency: int
    warmup_queries: int
    index_timeout_seconds: float


@dataclass(frozen=True)
class QueryMeasurement:
    lambda_rag: float
    strategy: str
    query_index: int
    recall_at_k: float
    latency_ms: float
    effective_horizon: int
    candidate_hits: int
    database_calls: int


@dataclass(frozen=True)
class SearchOutcome:
    results: list[WeightedCandidate]
    effective_horizon: int
    candidate_hits: int
    database_calls: int


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact and Qdrant-backed temporal retrieval strategies "
            "against iRAG's weighted top-K objective."
        )
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant-grpc-port", type=int, default=6334)
    parser.add_argument("--collection-name", default="irag_temporal_benchmark")
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--points", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--cluster-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-threshold", type=float, default=0.7)
    parser.add_argument(
        "--lambdas",
        default="1,0.9995,0.99861,0.995,0.99",
        help="Comma-separated retrieval-decay values.",
    )
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument(
        "--certified-scan-block",
        type=int,
        default=32,
        help="Vectors evaluated per newest-first exact-scan block.",
    )
    parser.add_argument("--fixed-window", type=int, default=500)
    parser.add_argument("--initial-window", type=int, default=250)
    parser.add_argument("--window-growth", type=float, default=2.0)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construct", type=int, default=128)
    parser.add_argument("--hnsw-ef", type=int, default=128)
    parser.add_argument(
        "--full-scan-threshold",
        type=int,
        default=10,
        help="Qdrant's minimum legal threshold, forcing ANN in useful windows.",
    )
    parser.add_argument("--indexing-threshold", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup-queries", type=int, default=50)
    parser.add_argument("--index-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    lambdas = tuple(float(value.strip()) for value in args.lambdas.split(","))
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            REPOSITORY_ROOT
            / "outputs"
            / f"temporal-retrieval-{timestamp}-{uuid.uuid4().hex[:8]}"
        )
    config = BenchmarkConfig(
        qdrant_url=args.qdrant_url,
        qdrant_grpc_port=args.qdrant_grpc_port,
        collection_name=args.collection_name,
        embedding_dir=args.embedding_dir.resolve(),
        output_dir=output_dir.resolve(),
        points=args.points,
        queries=args.queries,
        seed=args.seed,
        noise_std=args.noise_std,
        cluster_size=args.cluster_size,
        top_k=args.top_k,
        semantic_threshold=args.semantic_threshold,
        lambdas=lambdas,
        candidate_limit=args.candidate_limit,
        certified_scan_block=args.certified_scan_block,
        fixed_window=args.fixed_window,
        initial_window=args.initial_window,
        window_growth=args.window_growth,
        hnsw_m=args.hnsw_m,
        ef_construct=args.ef_construct,
        hnsw_ef=args.hnsw_ef,
        full_scan_threshold=args.full_scan_threshold,
        indexing_threshold=args.indexing_threshold,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        warmup_queries=args.warmup_queries,
        index_timeout_seconds=args.index_timeout_seconds,
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    if config.points < config.top_k:
        raise ValueError("points must be at least top_k")
    if config.queries <= 0 or config.top_k <= 0:
        raise ValueError("queries and top_k must be positive")
    if not 0 <= config.semantic_threshold <= 1:
        raise ValueError("semantic_threshold must be in [0, 1]")
    if any(not 0 < value <= 1 for value in config.lambdas):
        raise ValueError("every lambda must be in (0, 1]")
    if config.candidate_limit < config.top_k:
        raise ValueError("candidate_limit must be at least top_k")
    if (
        config.fixed_window <= 0
        or config.initial_window <= 0
        or config.certified_scan_block <= 0
    ):
        raise ValueError("window sizes must be positive")
    if config.window_growth <= 1:
        raise ValueError("window_growth must exceed 1")
    if config.cluster_size < config.top_k:
        raise ValueError("cluster_size must be at least top_k")
    if config.batch_size <= 0 or config.concurrency <= 0:
        raise ValueError("batch_size and concurrency must be positive")
    if config.full_scan_threshold < 10:
        raise ValueError("full_scan_threshold must be at least 10")
    if config.indexing_threshold <= 0:
        raise ValueError("indexing_threshold must be positive")
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative")


def load_base_embeddings(embedding_dir: Path) -> np.ndarray:
    batches: list[np.ndarray] = []
    for quarter in ("Q1", "Q2", "Q3", "Q4"):
        path = embedding_dir / f"{quarter}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding archive: {path}")
        with np.load(path, allow_pickle=False) as archive:
            vectors = archive["embeddings"].astype(np.float32, copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.all(np.isfinite(norms)):
            raise ValueError(f"Invalid vector in {path}")
        batches.append(vectors / norms)
    return np.ascontiguousarray(np.vstack(batches), dtype=np.float32)


def build_scaled_workload(
    base_vectors: np.ndarray,
    *,
    points: int,
    queries: int,
    cluster_size: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Scale real embeddings into a deterministic clustered ANN workload."""

    rng = np.random.default_rng(seed)
    prototype_count = min(
        base_vectors.shape[0],
        max(queries, math.ceil(points / cluster_size)),
    )
    prototype_ids = rng.choice(
        base_vectors.shape[0],
        size=prototype_count,
        replace=False,
    )
    assignments = np.resize(prototype_ids, points)
    rng.shuffle(assignments)
    vectors = np.array(base_vectors[assignments], dtype=np.float32, copy=True)
    if noise_std:
        vectors += rng.standard_normal(vectors.shape, dtype=np.float32) * noise_std
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    query_prototypes = rng.choice(prototype_ids, size=queries, replace=True)
    query_vectors = np.array(
        base_vectors[query_prototypes],
        dtype=np.float32,
        copy=True,
    )
    metadata: dict[str, int | float] = {
        "source_vectors": int(base_vectors.shape[0]),
        "dimensions": int(base_vectors.shape[1]),
        "prototype_count": int(prototype_count),
        "target_cluster_size": cluster_size,
        "noise_std": noise_std,
    }
    return np.ascontiguousarray(vectors), query_vectors, metadata


def ordinary_hnsw_search(
    store: QdrantVectorStore,
    config: BenchmarkConfig,
    query_vector: np.ndarray,
    lambda_rag: float,
) -> SearchOutcome:
    candidates = store.query(
        collection_name=config.collection_name,
        query_vector=query_vector,
        limit=config.candidate_limit,
        semantic_threshold=config.semantic_threshold,
        hnsw_ef=config.hnsw_ef,
    )
    results = rerank_candidates(
        candidates,
        knowledge_base_size=config.points,
        lambda_rag=lambda_rag,
        semantic_threshold=config.semantic_threshold,
        top_k=config.top_k,
    )
    return SearchOutcome(results, config.points, len(candidates), 1)


def fixed_recency_search(
    store: QdrantVectorStore,
    config: BenchmarkConfig,
    query_vector: np.ndarray,
    lambda_rag: float,
) -> SearchOutcome:
    horizon = min(config.fixed_window, config.points)
    candidates = store.query(
        collection_name=config.collection_name,
        query_vector=query_vector,
        limit=min(config.candidate_limit, horizon),
        semantic_threshold=config.semantic_threshold,
        hnsw_ef=config.hnsw_ef,
        minimum_insertion_index=config.points - horizon + 1,
    )
    results = rerank_candidates(
        candidates,
        knowledge_base_size=config.points,
        lambda_rag=lambda_rag,
        semantic_threshold=config.semantic_threshold,
        top_k=config.top_k,
    )
    return SearchOutcome(results, horizon, len(candidates), 1)


def adaptive_temporal_search(
    store: QdrantVectorStore,
    config: BenchmarkConfig,
    query_vector: np.ndarray,
    lambda_rag: float,
) -> SearchOutcome:
    if lambda_rag == 1:
        candidates = store.query(
            collection_name=config.collection_name,
            query_vector=query_vector,
            limit=config.candidate_limit,
            semantic_threshold=config.semantic_threshold,
            hnsw_ef=config.hnsw_ef,
        )
        results = rerank_candidates(
            candidates,
            knowledge_base_size=config.points,
            lambda_rag=lambda_rag,
            semantic_threshold=config.semantic_threshold,
            top_k=config.top_k,
        )
        return SearchOutcome(results, config.points, len(candidates), 1)

    horizon = min(config.initial_window, config.points)
    total_candidates = 0
    calls = 0
    while True:
        minimum_index = None
        if horizon < config.points:
            minimum_index = config.points - horizon + 1
        candidates = store.query(
            collection_name=config.collection_name,
            query_vector=query_vector,
            limit=min(config.candidate_limit, horizon),
            semantic_threshold=config.semantic_threshold,
            hnsw_ef=config.hnsw_ef,
            minimum_insertion_index=minimum_index,
        )
        calls += 1
        total_candidates += len(candidates)
        results = rerank_candidates(
            candidates,
            knowledge_base_size=config.points,
            lambda_rag=lambda_rag,
            semantic_threshold=config.semantic_threshold,
            top_k=config.top_k,
        )
        kth_score = results[-1].weighted_score if len(results) >= config.top_k else 0
        if horizon == config.points or temporal_pruning_is_safe(
            lambda_rag=lambda_rag,
            searched_horizon=horizon,
            kth_score=kth_score,
            result_count=len(results),
            top_k=config.top_k,
        ):
            return SearchOutcome(results, horizon, total_candidates, calls)
        horizon = min(
            config.points,
            max(horizon + 1, math.ceil(horizon * config.window_growth)),
        )


def exact_search(
    vectors: np.ndarray,
    config: BenchmarkConfig,
    query_vector: np.ndarray,
    lambda_rag: float,
) -> SearchOutcome:
    results = exhaustive_weighted_top_k(
        vectors,
        query_vector,
        lambda_rag=lambda_rag,
        semantic_threshold=config.semantic_threshold,
        top_k=config.top_k,
    )
    return SearchOutcome(results, config.points, config.points, 0)


def certified_scan_search(
    vectors: np.ndarray,
    config: BenchmarkConfig,
    query_vector: np.ndarray,
    lambda_rag: float,
) -> SearchOutcome:
    """Run the exact newest-to-oldest scan stopped by the certificate."""

    results, horizon = certified_newest_first_top_k(
        vectors,
        query_vector,
        lambda_rag=lambda_rag,
        semantic_threshold=config.semantic_threshold,
        top_k=config.top_k,
        block_size=config.certified_scan_block,
    )
    return SearchOutcome(results, horizon, horizon, 0)


def timed_search(search: Callable[[], SearchOutcome]) -> tuple[SearchOutcome, float]:
    start = time.perf_counter_ns()
    outcome = search()
    latency_ms = (time.perf_counter_ns() - start) / 1_000_000
    return outcome, latency_ms


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def benchmark_strategy(
    *,
    strategy_name: str,
    search: Callable[[np.ndarray], SearchOutcome],
    query_vectors: np.ndarray,
    truth: list[list[WeightedCandidate]],
    lambda_rag: float,
    warmup_queries: int,
    concurrency: int,
) -> tuple[list[QueryMeasurement], float]:
    for query_vector in query_vectors[:warmup_queries]:
        search(query_vector)

    measurements: list[QueryMeasurement] = []
    for query_index, query_vector in enumerate(query_vectors):
        outcome, latency_ms = timed_search(lambda: search(query_vector))
        measurements.append(
            QueryMeasurement(
                lambda_rag=lambda_rag,
                strategy=strategy_name,
                query_index=query_index,
                recall_at_k=recall_at_k(
                    (candidate.point_id for candidate in truth[query_index]),
                    (candidate.point_id for candidate in outcome.results),
                ),
                latency_ms=latency_ms,
                effective_horizon=outcome.effective_horizon,
                candidate_hits=outcome.candidate_hits,
                database_calls=outcome.database_calls,
            )
        )

    throughput_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(search, query_vectors))
    throughput_seconds = time.perf_counter() - throughput_start
    throughput = len(query_vectors) / throughput_seconds
    return measurements, throughput


def summarize_measurements(
    measurements: Sequence[QueryMeasurement],
    throughput: float,
    points: int,
) -> dict[str, float | int | str]:
    recalls = [measurement.recall_at_k for measurement in measurements]
    latencies = [measurement.latency_ms for measurement in measurements]
    horizons = [measurement.effective_horizon for measurement in measurements]
    candidates = [measurement.candidate_hits for measurement in measurements]
    calls = [measurement.database_calls for measurement in measurements]
    first = measurements[0]
    return {
        "lambda_rag": first.lambda_rag,
        "strategy": first.strategy,
        "queries": len(measurements),
        "mean_recall_at_k": statistics.fmean(recalls),
        "recall_std": statistics.pstdev(recalls),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "throughput_qps": throughput,
        "mean_effective_horizon": statistics.fmean(horizons),
        "p50_effective_horizon": percentile(horizons, 50),
        "p95_effective_horizon": percentile(horizons, 95),
        "mean_horizon_fraction": statistics.fmean(horizons) / points,
        "mean_candidate_hits": statistics.fmean(candidates),
        "mean_database_calls": statistics.fmean(calls),
    }


def ingest_workload(
    store: QdrantVectorStore,
    config: BenchmarkConfig,
    vectors: np.ndarray,
) -> dict[str, float | int | list[float]]:
    batch_latencies: list[float] = []
    ingest_start = time.perf_counter()
    for start in range(0, config.points, config.batch_size):
        end = min(start + config.batch_size, config.points)
        batch_start = time.perf_counter()
        store.upsert_batch(
            collection_name=config.collection_name,
            point_ids=list(range(start + 1, end + 1)),
            insertion_indices=list(range(start + 1, end + 1)),
            vectors=vectors[start:end],
        )
        batch_latencies.append((time.perf_counter() - batch_start) * 1_000)
        if end == config.points or end % max(config.batch_size, 1_000) == 0:
            LOGGER.info("Uploaded %s/%s vectors", end, config.points)
    ingest_seconds = time.perf_counter() - ingest_start

    indexing_start = time.perf_counter()
    status = store.wait_until_indexed(
        collection_name=config.collection_name,
        expected_points=config.points,
        timeout_seconds=config.index_timeout_seconds,
    )
    indexing_wait_seconds = time.perf_counter() - indexing_start
    return {
        "batch_count": len(batch_latencies),
        "batch_size": config.batch_size,
        "total_ingest_seconds": ingest_seconds,
        "vectors_per_second": config.points / ingest_seconds,
        "p50_batch_latency_ms": percentile(batch_latencies, 50),
        "p95_batch_latency_ms": percentile(batch_latencies, 95),
        "indexing_wait_seconds": indexing_wait_seconds,
        "indexed_vectors_count": status.indexed_vectors_count,
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: Sequence[dict[str, object]], output_path: Path) -> None:
    strategies = (
        "exhaustive_weighted",
        "certified_scan",
        "ordinary_hnsw",
        "fixed_recency",
        "adaptive_temporal",
    )
    labels = {
        "exhaustive_weighted": "Exhaustive weighted",
        "certified_scan": "Certified exact scan",
        "ordinary_hnsw": "Ordinary HNSW",
        "fixed_recency": "Fixed recency",
        "adaptive_temporal": "Adaptive temporal",
    }
    lambdas = sorted({float(row["lambda_rag"]) for row in summary_rows})
    x = np.arange(len(lambdas))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    panels = (
        ("mean_recall_at_k", "Weighted-retrieval recall@K"),
        ("p95_latency_ms", "p95 latency (ms)"),
        ("throughput_qps", "Throughput (queries/s)"),
        ("mean_horizon_fraction", "Mean effective horizon (fraction of KB)"),
    )
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for strategy in strategies:
            values = [
                float(
                    next(
                        row[metric]
                        for row in summary_rows
                        if row["strategy"] == strategy
                        and float(row["lambda_rag"]) == lambda_rag
                    )
                )
                for lambda_rag in lambdas
            ]
            axis.plot(x, values, marker="o", linewidth=1.8, label=labels[strategy])
        axis.set_title(title)
        axis.set_xticks(x, [f"{value:g}" for value in lambdas], rotation=20)
        axis.set_xlabel(r"$\lambda_{\mathrm{RAG}}$")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.02)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Temporal-Retrieval Benchmark", fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(
    summary_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$\lambda_{\mathrm{RAG}}$ & Strategy & Recall@$K$ & p50 ms & p95 ms & Horizon \\" ,
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{float(row['lambda_rag']):g} & "
            f"{str(row['strategy']).replace('_', r'\_')} & "
            f"{float(row['mean_recall_at_k']):.3f} & "
            f"{float(row['p50_latency_ms']):.2f} & "
            f"{float(row['p95_latency_ms']):.2f} & "
            f"{100 * float(row['mean_horizon_fraction']):.1f}\\% \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def serializable_config(config: BenchmarkConfig) -> dict[str, object]:
    data = asdict(config)
    for field_name, path in (
        ("embedding_dir", config.embedding_dir),
        ("output_dir", config.output_dir),
    ):
        try:
            data[field_name] = str(path.relative_to(REPOSITORY_ROOT))
        except ValueError:
            data[field_name] = str(path)
    data["lambdas"] = list(config.lambdas)
    return data


def run(config: BenchmarkConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading source embeddings from %s", config.embedding_dir)
    base_vectors = load_base_embeddings(config.embedding_dir)
    vectors, query_vectors, workload_metadata = build_scaled_workload(
        base_vectors,
        points=config.points,
        queries=config.queries,
        cluster_size=config.cluster_size,
        noise_std=config.noise_std,
        seed=config.seed,
    )
    matrix_memory_bytes = int(vectors.nbytes)

    store = QdrantVectorStore(
        url=config.qdrant_url,
        grpc_port=config.qdrant_grpc_port,
    )
    try:
        try:
            store.wait_until_ready()
        except TimeoutError as exc:
            raise RuntimeError(
                f"Qdrant is not ready at {config.qdrant_url}; run "
                "`make vector-db-up` first."
            ) from exc
        qdrant_version = store.version()
        rss_before = store.resident_memory_bytes()
        LOGGER.info("Creating Qdrant collection %s", config.collection_name)
        store.recreate_collection(
            collection_name=config.collection_name,
            dimension=vectors.shape[1],
            hnsw_m=config.hnsw_m,
            ef_construct=config.ef_construct,
            full_scan_threshold=config.full_scan_threshold,
            indexing_threshold=config.indexing_threshold,
        )
        update_metrics = ingest_workload(store, config, vectors)
        rss_after = store.resident_memory_bytes()
        collection_memory = store.collection_memory(config.collection_name)

        all_measurements: list[QueryMeasurement] = []
        summary_rows: list[dict[str, object]] = []
        for lambda_rag in config.lambdas:
            LOGGER.info("Computing exact ground truth for lambda=%s", lambda_rag)
            truth = [
                exhaustive_weighted_top_k(
                    vectors,
                    query_vector,
                    lambda_rag=lambda_rag,
                    semantic_threshold=config.semantic_threshold,
                    top_k=config.top_k,
                )
                for query_vector in query_vectors
            ]
            strategies: tuple[
                tuple[str, Callable[[np.ndarray], SearchOutcome]], ...
            ] = (
                (
                    "exhaustive_weighted",
                    lambda query, decay=lambda_rag: exact_search(
                        vectors, config, query, decay
                    ),
                ),
                (
                    "certified_scan",
                    lambda query, decay=lambda_rag: certified_scan_search(
                        vectors, config, query, decay
                    ),
                ),
                (
                    "ordinary_hnsw",
                    lambda query, decay=lambda_rag: ordinary_hnsw_search(
                        store, config, query, decay
                    ),
                ),
                (
                    "fixed_recency",
                    lambda query, decay=lambda_rag: fixed_recency_search(
                        store, config, query, decay
                    ),
                ),
                (
                    "adaptive_temporal",
                    lambda query, decay=lambda_rag: adaptive_temporal_search(
                        store, config, query, decay
                    ),
                ),
            )
            for strategy_name, search in strategies:
                LOGGER.info(
                    "Benchmarking %s at lambda=%s", strategy_name, lambda_rag
                )
                measurements, throughput = benchmark_strategy(
                    strategy_name=strategy_name,
                    search=search,
                    query_vectors=query_vectors,
                    truth=truth,
                    lambda_rag=lambda_rag,
                    warmup_queries=min(config.warmup_queries, config.queries),
                    concurrency=config.concurrency,
                )
                all_measurements.extend(measurements)
                summary_rows.append(
                    summarize_measurements(measurements, throughput, config.points)
                )
    finally:
        store.close()

    measurement_rows = [asdict(measurement) for measurement in all_measurements]
    write_csv(config.output_dir / "per-query.csv", measurement_rows)
    write_csv(config.output_dir / "summary.csv", summary_rows)
    plot_summary(summary_rows, config.output_dir / "benchmark.png")
    write_latex_table(summary_rows, config.output_dir / "benchmark-table.tex")
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "qdrant_client": version("qdrant-client"),
        },
        "config": serializable_config(config),
        "workload": {
            **workload_metadata,
            "matrix_memory_bytes": matrix_memory_bytes,
        },
        "qdrant": {
            "version": qdrant_version,
            "resident_memory_before_bytes": rss_before,
            "resident_memory_after_bytes": rss_after,
            "resident_memory_delta_bytes": (
                None
                if rss_before is None or rss_after is None
                else rss_after - rss_before
            ),
            "collection_memory": collection_memory,
        },
        "index_update": update_metrics,
        "summary": summary_rows,
        "artifacts": {
            "per_query_csv": "per-query.csv",
            "summary_csv": "summary.csv",
            "plot": "benchmark.png",
            "latex_table": "benchmark-table.tex",
        },
    }
    (config.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("Benchmark artifacts written to %s", config.output_dir)
    return config.output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    output_dir = run(parse_args())
    print(output_dir)


if __name__ == "__main__":
    main()
