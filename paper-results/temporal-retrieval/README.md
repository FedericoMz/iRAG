# Temporal-retrieval benchmark

This directory is the complete result bundle for the 50,000-vector temporal-
retrieval benchmark reported in the paper. The five strategies are:

- `exhaustive_weighted`: exact NumPy evaluation of the weighted objective and
  the recall ground truth;
- `certified_scan`: exact newest-to-oldest block scan, stopped only when the
  temporal certificate excludes every unscanned record;
- `ordinary_hnsw`: global semantic HNSW candidates followed by exact temporal
  reranking;
- `fixed_recency`: the same HNSW search restricted to the latest 500 records;
- `adaptive_temporal`: geometrically expanding metadata-filtered HNSW windows,
  stopped only when the temporal certificate holds or the full KB is reached.

Files:

- `result.json`: full configuration, workload metadata, Qdrant version,
  storage/memory, index-update measurements, and aggregate results;
- `per-query.csv`: recall, client-observed latency, effective horizon,
  candidate count, and database-call count for every strategy, lambda, and
  query;
- `summary.csv`: recall, p50/p95 latency, four-client throughput, and horizon
  summaries;
- `benchmark.png`: four-panel result plot;
- `benchmark-table.tex`: compact generated LaTeX table.

Reproduce the workload after `make install` with:

```sh
make paper-temporal-benchmark
```

The seed is `20260717`; source embeddings are the checked-in normalised
`qwen3-embedding:4b` ticket embeddings. No model or embedding API is called.
The 50,000-vector collection is a deterministic clustered expansion of those
2,000 embeddings. The benchmark uses one HNSW candidate budget and one
four-client concurrency setting; it does not establish general performance
across corpus sizes, embedding collections, or internal threading regimes.
