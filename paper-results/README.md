# Paper statistical artifacts

These files reproduce the statistical reporting in Section 4 of the paper.
Regenerate them from the four completed experiment jobs with:

```bash
make paper-stats
```

Files:

- `per-run-results.csv`: all 40 individual runs, including seeds, rates,
  integer event counts, eligible denominators, profile-assignment counts,
  order and assignment fingerprints, final FEA, and final state.
- `condition-summary.csv`: mean, sample standard deviation, two-sided 95%
  Student-t confidence interval, and range for every reported condition-level
  metric and count.
- `paired-comparisons.csv`: matched coupled-decay-minus-no-decay differences
  for the four primary outcomes, with paired confidence intervals, Cohen's dz,
  exact sign-flip p-values, and Holm-adjusted p-values.
- `baseline-comparisons.csv`: matched final-error-minus-human-baseline
  comparisons for all four conditions.
- `controller-comparisons.csv`: matched raw-LLM-error-minus-final-error
  comparisons, quantifying the immediate contribution of the SO/SC/DS
  authority controller while holding each realised retrieval trajectory fixed.
- `paper-statistics.json`: the same summaries plus definitions, paired
  differences, final-state frequencies, and source job IDs.

The primary outcomes are final-decision error, stable-ticket error,
drift-ticket error, and model-finalized proportion. Rates and paired
differences in the CSV files are expressed in percentage points; FEA remains
on its native 0--1 scale. In these artifacts, D uses
`lambda_rag=lambda_fea=0.99861`, while ND uses
`lambda_rag=lambda_fea=1`. D and ND runs are paired by seed and have identical
ticket order and human assignments within each drift setting.
The export fails if either fingerprint differs within a paired comparison.
