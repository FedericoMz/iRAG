# SalesX 40% quarterly-drift variant

This dataset is the high-drift counterpart of the SalesX benchmark in the
parent directory.

- Q1 is the canonical 500-ticket baseline stored in `../shared`; it is the
  exact same input used by the 10% variant.
- Q2, Q3, and Q4 each contain 200 changed-answer tickets (40%), produced by
  eight changed articles in each of the five categories.
- Record IDs, question text, shuffle order, category, and difficulty are
  identical to the 10% benchmark. Only the quarterly product behavior,
  gold/profile answers, drift lineage, and documentation are changed.
- Each of the 120 quarter/article changes has its own authored rule and change
  dimensions. All 100 articles change at least once; only the minimum 20
  articles needed to reach 120 change events receive a second evolution.
- As in the original benchmark, an article's five question variants share
  answers by difficulty: two easy questions reuse one answer, two normal
  questions reuse one answer, and the hard question has an expanded answer.
- The 50-ticket `Extra` abstention split is also loaded from `../shared`.

Because embeddings are computed only from the question field, this variant
reuses the checked-in embeddings in the parent directory. The dataset loader
checks every variant ID/question pair against the embedding source before an
experiment starts.

Generate and validate from the repository root:

```sh
python3 "experiment data/generate_drift_40_dataset.py"
python3 "experiment data/validate_drift_40_dataset.py"
```

Select this dataset with `dataset=drift_40` in the API or with the
corresponding dropdown in `/docs`.
