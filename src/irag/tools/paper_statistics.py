#!/usr/bin/env python3
"""Export matched-seed statistics for the factorial paper experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import shutil
import statistics
from pathlib import Path

from irag.core.config import settings
from irag.tools.analyze_experiment import (
    find_job_directory,
    load_completed_runs,
    plot_average_results,
)
from irag.tools.compose_plot_grid import compose_plot_grid
from irag.tools.plot_experiment_result import static_rag_defer_error
from irag.tools.plot_legend import plot_legend


PAPER_JOBS = {
    ("10", "D"): "45db2764a46e4506ba34886786ad7983",
    ("10", "R_D_F_ND"): "d35eae33e3b04c90a2f35f8959151719",
    ("10", "R_ND_F_D"): "e5b37af13f4d47308fb8e18d967c0333",
    ("10", "ND"): "030f145d48d844d49a2872c805ace823",
    ("40", "D"): "1ad12d351a8b4e7391a286ed1d414c65",
    ("40", "R_D_F_ND"): "7d41af2df5a5489e8fe0429144459dfb",
    ("40", "R_ND_F_D"): "bb5b89fed1e84bd5bd4108b9e3d3e9a9",
    ("40", "ND"): "4e9394df9ba3463db13bc3a604d37afc",
}

CONDITION_SLUGS = {
    "D": "rag-d__fea-d",
    "R_D_F_ND": "rag-d__fea-nd",
    "R_ND_F_D": "rag-nd__fea-d",
    "ND": "rag-nd__fea-nd",
}

PLOT_ORDER = (
    ("10", "D"),
    ("40", "D"),
    ("10", "R_D_F_ND"),
    ("40", "R_D_F_ND"),
    ("10", "R_ND_F_D"),
    ("40", "R_ND_F_D"),
    ("10", "ND"),
    ("40", "ND"),
)

STATE_NAMES = {
    "SO": "silent_observer",
    "SC": "skeptical_contestator",
    "DS": "deferring_surrogate",
}

RATE_FIELDS = (
    "human_only_error_pct",
    "llm_gold_error_pct",
    "static_rag_defer_error_pct",
    "final_error_pct",
    "stable_error_pct",
    "drift_error_pct",
    "abstain_so_pct",
    "abstain_sc_pct",
    "abstain_ds_pct",
    "sc_acceptance_pct",
    "model_finalized_pct",
)

PRIMARY_FIELDS = (
    "final_error_pct",
    "stable_error_pct",
    "drift_error_pct",
    "model_finalized_pct",
)

# Two-sided 95% Student-t critical value for the paper's ten repetitions.
T_CRITICAL_DF_9 = 2.2621571627409915


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-run and matched statistics for the paper jobs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.output_dir,
        help="Root containing completed experiment folders.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("paper-results"),
        help="Directory receiving CSV and JSON statistical artifacts.",
    )
    return parser.parse_args()


def count_and_rate(tickets: list[dict], predicate) -> tuple[int, int, float]:
    numerator = sum(int(bool(predicate(ticket))) for ticket in tickets)
    denominator = len(tickets)
    rate = numerator / denominator if denominator else math.nan
    return numerator, denominator, rate


def per_run_metrics(run: dict, drift_rate: str, decay: str) -> dict:
    tickets = [
        ticket for ticket in run["tickets"] if ticket["quarter"] != "Extra"
    ]
    row = {
        "drift_rate": int(drift_rate),
        "decay": decay,
        "repetition": int(run["repetition"]),
        "seed": int(run["seed"]),
        "tickets": len(tickets),
        "ticket_order_sha256": hashlib.sha256(
            "\n".join(ticket["ticket_id"] for ticket in tickets).encode()
        ).hexdigest(),
        "human_assignment_sha256": hashlib.sha256(
            "\n".join(
                f'{ticket["ticket_id"]}|{ticket["assigned_profile"]}'
                for ticket in tickets
            ).encode()
        ).hexdigest(),
    }

    for profile in ("ceo", "domain_expert", "intern"):
        row[f"assigned_{profile}_count"] = sum(
            ticket["assigned_profile"] == profile for ticket in tickets
        )

    scopes = {
        "human_only_error": (
            tickets,
            lambda ticket: not ticket["human_answer_is_correct"],
        ),
        "final_error": (
            tickets,
            lambda ticket: ticket["final_decision_error"],
        ),
        "llm_gold_error": (
            [
                ticket
                for ticket in tickets
                if ticket.get("gold_judgment") is not None
                and not bool((ticket.get("model_decision") or {}).get("abstain"))
            ],
            lambda ticket: not bool(
                ticket["gold_judgment"]["gold_reference_covered"]
            ),
        ),
        "static_rag_defer_error": (
            tickets,
            static_rag_defer_error,
        ),
        "stable_error": (
            [ticket for ticket in tickets if not ticket["is_drift"]],
            lambda ticket: ticket["final_decision_error"],
        ),
        "drift_error": (
            [ticket for ticket in tickets if ticket["is_drift"]],
            lambda ticket: ticket["final_decision_error"],
        ),
        "model_finalized": (
            tickets,
            lambda ticket: ticket["metric_component"] == "autonomous_model",
        ),
    }
    for name, (scope, predicate) in scopes.items():
        count, denominator, rate = count_and_rate(scope, predicate)
        row[f"{name}_count"] = count
        row[f"{name}_denominator"] = denominator
        row[f"{name}_pct"] = 100 * rate

    for short_name, state_name in STATE_NAMES.items():
        scope = [
            ticket for ticket in tickets if ticket["state_before"] == state_name
        ]
        count, denominator, rate = count_and_rate(
            scope,
            lambda ticket: ticket["model_decision"]["abstain"],
        )
        key = short_name.lower()
        row[f"state_{key}_tickets"] = denominator
        row[f"abstain_{key}_count"] = count
        row[f"abstain_{key}_pct"] = 100 * rate

    sc_decisions = [
        ticket
        for ticket in tickets
        if ticket["state_before"] == STATE_NAMES["SC"]
        and isinstance(ticket.get("suggestion_accepted"), bool)
    ]
    accepted, opportunities, acceptance_rate = count_and_rate(
        sc_decisions,
        lambda ticket: ticket["suggestion_accepted"],
    )
    row["sc_accepted_count"] = accepted
    row["sc_acceptance_opportunities"] = opportunities
    row["sc_acceptance_pct"] = 100 * acceptance_rate
    row["final_fea"] = float(tickets[-1]["fea_after"])
    row["final_state"] = tickets[-1]["state_after"]
    row["reliability_observations"] = int(tickets[-1]["observations_after"])
    return row


def summarize(values: list[float]) -> dict:
    if len(values) != 10:
        raise ValueError("Paper statistics require exactly ten repetitions")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    half_width = T_CRITICAL_DF_9 * sample_sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "minimum": min(values),
        "maximum": max(values),
    }


def exact_sign_flip_p(differences: list[float]) -> float:
    """Two-sided exact paired randomization p-value for the mean difference."""
    observed = abs(statistics.fmean(differences))
    extreme = 0
    assignments = 2 ** len(differences)
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        permuted = abs(
            sum(sign * difference for sign, difference in zip(signs, differences))
            / len(differences)
        )
        if permuted >= observed - 1e-12:
            extreme += 1
    return extreme / assignments


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    running_maximum = 0.0
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def paired_comparisons(condition_rows: dict[tuple[str, str], list[dict]]) -> list[dict]:
    comparisons = []
    for drift_rate in ("10", "40"):
        decay_by_seed = {
            row["seed"]: row for row in condition_rows[(drift_rate, "D")]
        }
        no_decay_by_seed = {
            row["seed"]: row for row in condition_rows[(drift_rate, "ND")]
        }
        if decay_by_seed.keys() != no_decay_by_seed.keys():
            raise ValueError(f"Seeds are not paired for {drift_rate}% drift")
        for seed in decay_by_seed:
            for fingerprint in (
                "ticket_order_sha256",
                "human_assignment_sha256",
            ):
                if decay_by_seed[seed][fingerprint] != no_decay_by_seed[seed][
                    fingerprint
                ]:
                    raise ValueError(
                        f"Mismatched {fingerprint} for {drift_rate}% drift, "
                        f"seed {seed}"
                    )

        metric_rows = []
        raw_p_values = []
        for metric in PRIMARY_FIELDS:
            differences = [
                decay_by_seed[seed][metric] - no_decay_by_seed[seed][metric]
                for seed in sorted(decay_by_seed)
            ]
            summary = summarize(differences)
            raw_p = exact_sign_flip_p(differences)
            raw_p_values.append(raw_p)
            metric_rows.append(
                {
                    "drift_rate": int(drift_rate),
                    "metric": metric,
                    "difference_direction": "D_minus_ND",
                    **summary,
                    "cohens_dz": (
                        summary["mean"] / summary["sample_sd"]
                        if summary["sample_sd"]
                        else None
                    ),
                    "exact_sign_flip_p": raw_p,
                    "paired_differences": differences,
                }
            )
        for row, adjusted_p in zip(
            metric_rows,
            holm_adjust(raw_p_values),
            strict=True,
        ):
            row["holm_adjusted_p"] = adjusted_p
            comparisons.append(row)
    return comparisons


def factorial_comparisons(
    condition_rows: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    """Compare each decay factor while holding the other factor fixed."""
    contrasts = (
        ("RAG", "FEA_D", "D", "R_ND_F_D"),
        ("RAG", "FEA_ND", "R_D_F_ND", "ND"),
        ("FEA", "RAG_D", "D", "R_D_F_ND"),
        ("FEA", "RAG_ND", "R_ND_F_D", "ND"),
    )
    comparisons = []
    for drift_rate in ("10", "40"):
        for factor, fixed_factor, decay_condition, reference_condition in contrasts:
            decay_by_seed = {
                row["seed"]: row
                for row in condition_rows[(drift_rate, decay_condition)]
            }
            reference_by_seed = {
                row["seed"]: row
                for row in condition_rows[(drift_rate, reference_condition)]
            }
            if decay_by_seed.keys() != reference_by_seed.keys():
                raise ValueError(
                    f"Seeds are not paired for {drift_rate}% drift, "
                    f"{factor} with {fixed_factor}"
                )
            for seed in decay_by_seed:
                for fingerprint in (
                    "ticket_order_sha256",
                    "human_assignment_sha256",
                ):
                    if decay_by_seed[seed][fingerprint] != reference_by_seed[seed][
                        fingerprint
                    ]:
                        raise ValueError(
                            f"Mismatched {fingerprint} for {drift_rate}% drift, "
                            f"{factor} with {fixed_factor}, seed {seed}"
                        )

            block = []
            raw_p_values = []
            for metric in PRIMARY_FIELDS:
                differences = [
                    decay_by_seed[seed][metric]
                    - reference_by_seed[seed][metric]
                    for seed in sorted(decay_by_seed)
                ]
                summary = summarize(differences)
                raw_p = exact_sign_flip_p(differences)
                raw_p_values.append(raw_p)
                block.append(
                    {
                        "drift_rate": int(drift_rate),
                        "factor": factor,
                        "fixed_factor": fixed_factor,
                        "decay_condition": decay_condition,
                        "reference_condition": reference_condition,
                        "metric": metric,
                        "difference_direction": (
                            f"{decay_condition}_minus_{reference_condition}"
                        ),
                        **summary,
                        "cohens_dz": (
                            summary["mean"] / summary["sample_sd"]
                            if summary["sample_sd"]
                            else None
                        ),
                        "exact_sign_flip_p": raw_p,
                        "paired_differences": differences,
                    }
                )
            for row, adjusted_p in zip(
                block,
                holm_adjust(raw_p_values),
                strict=True,
            ):
                row["holm_adjusted_p"] = adjusted_p
                comparisons.append(row)
    return comparisons


def normalized_answer(value: str | None) -> str:
    return " ".join((value or "").split())


def load_gold_answers(data_dir: Path, drift_rate: str) -> dict[str, str]:
    paths = [
        data_dir / "shared" / "Q1_qa.json",
        *(data_dir / f"drift_{drift_rate}").glob("Q[234]_qa.json"),
    ]
    answers = {}
    for path in paths:
        records = json.loads(path.read_text(encoding="utf-8"))
        answers.update(
            {record["id"]: record["gold_answer"] for record in records}
        )
    return answers


def auxiliary_judge_audit(
    condition_runs: dict[tuple[str, str], list[dict]],
    data_dir: Path,
) -> dict:
    """Audit reference judgments used to discuss possible M/M_A self-bias."""
    counts = {
        "recorded_ticket_traces": 0,
        "non_abstaining_gold_judgments": 0,
        "gold_judgments_rejected": 0,
        "explicit_sc_decisions": 0,
        "explicit_sc_rejections": 0,
        "exact_incorrect_human_matches": 0,
        "exact_incorrect_human_matches_rejected": 0,
        "exact_gold_matches": 0,
        "exact_gold_matches_accepted": 0,
    }
    gold_by_drift = {
        drift_rate: load_gold_answers(data_dir, drift_rate)
        for drift_rate in ("10", "40")
    }
    for (drift_rate, _), runs in condition_runs.items():
        gold_answers = gold_by_drift[drift_rate]
        for run in runs:
            for ticket in run["tickets"]:
                if ticket["quarter"] == "Extra":
                    continue
                counts["recorded_ticket_traces"] += 1
                decision = ticket.get("model_decision") or {}
                if decision.get("abstain"):
                    continue

                judgment = ticket.get("gold_judgment")
                if judgment is not None:
                    counts["non_abstaining_gold_judgments"] += 1
                    if not judgment["gold_reference_covered"]:
                        counts["gold_judgments_rejected"] += 1

                accepted = ticket.get("suggestion_accepted")
                if isinstance(accepted, bool):
                    counts["explicit_sc_decisions"] += 1
                    if not accepted:
                        counts["explicit_sc_rejections"] += 1

                model_answer = normalized_answer(decision.get("answer"))
                if (
                    not ticket["human_answer_is_correct"]
                    and model_answer == normalized_answer(ticket["human_answer"])
                ):
                    counts["exact_incorrect_human_matches"] += 1
                    if judgment is not None and not judgment["gold_reference_covered"]:
                        counts["exact_incorrect_human_matches_rejected"] += 1

                if model_answer == normalized_answer(
                    gold_answers[ticket["ticket_id"]]
                ):
                    counts["exact_gold_matches"] += 1
                    if judgment is not None and judgment["gold_reference_covered"]:
                        counts["exact_gold_matches_accepted"] += 1

    def rate(numerator: str, denominator: str) -> float:
        return 100 * counts[numerator] / counts[denominator]

    return {
        "counts": counts,
        "rates_pct": {
            "explicit_sc_rejection": rate(
                "explicit_sc_rejections", "explicit_sc_decisions"
            ),
            "gold_judgment_rejection": rate(
                "gold_judgments_rejected", "non_abstaining_gold_judgments"
            ),
            "exact_incorrect_human_match_rejection": rate(
                "exact_incorrect_human_matches_rejected",
                "exact_incorrect_human_matches",
            ),
            "exact_gold_match_acceptance": rate(
                "exact_gold_matches_accepted", "exact_gold_matches"
            ),
        },
    }


def baseline_comparisons(condition_rows: dict[tuple[str, str], list[dict]]) -> list[dict]:
    comparisons = []
    raw_p_values = []
    for (drift_rate, decay), rows in condition_rows.items():
        differences = [
            row["final_error_pct"] - row["human_only_error_pct"]
            for row in rows
        ]
        summary = summarize(differences)
        raw_p = exact_sign_flip_p(differences)
        raw_p_values.append(raw_p)
        comparisons.append(
            {
                "drift_rate": int(drift_rate),
                "decay": decay,
                "difference_direction": "final_error_minus_human_only_error",
                **summary,
                "cohens_dz": (
                    summary["mean"] / summary["sample_sd"]
                    if summary["sample_sd"]
                    else None
                ),
                "exact_sign_flip_p": raw_p,
                "paired_differences": differences,
            }
        )
    for row, adjusted_p in zip(
        comparisons,
        holm_adjust(raw_p_values),
        strict=True,
    ):
        row["holm_adjusted_p"] = adjusted_p
    return comparisons


def controller_comparisons(
    condition_rows: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    comparisons = []
    raw_p_values = []
    for (drift_rate, decay), rows in condition_rows.items():
        differences = [
            row["static_rag_defer_error_pct"] - row["final_error_pct"]
            for row in rows
        ]
        summary = summarize(differences)
        raw_p = exact_sign_flip_p(differences)
        raw_p_values.append(raw_p)
        comparisons.append(
            {
                "drift_rate": int(drift_rate),
                "decay": decay,
                "difference_direction": (
                    "static_rag_defer_error_minus_final_error"
                ),
                **summary,
                "cohens_dz": (
                    summary["mean"] / summary["sample_sd"]
                    if summary["sample_sd"]
                    else None
                ),
                "exact_sign_flip_p": raw_p,
                "paired_differences": differences,
            }
        )
    for row, adjusted_p in zip(
        comparisons,
        holm_adjust(raw_p_values),
        strict=True,
    ):
        row["holm_adjusted_p"] = adjusted_p
    return comparisons


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in fieldnames} for row in rows
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compress_json(source: Path, destination: Path) -> None:
    """Losslessly compress a JSON artifact with deterministic gzip metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open("rb") as source_handle,
        destination.open("wb") as destination_handle,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=destination_handle,
            mtime=0,
        ) as compressed_handle,
    ):
        shutil.copyfileobj(source_handle, compressed_handle)


def export_source_artifacts(
    destination: Path,
    condition_results: dict[tuple[str, str], dict],
    condition_runs: dict[tuple[str, str], list[dict]],
    job_directories: dict[tuple[str, str], Path],
) -> tuple[list[Path], list[dict]]:
    """Export compact source-job evidence and plots for all paper conditions."""
    exported_paths: list[Path] = []
    manifest_rows: list[dict] = []
    plot_paths: dict[tuple[str, str], Path] = {}
    source_root = destination / "source-outputs"
    plot_root = destination / "plots"

    for condition, job_id in PAPER_JOBS.items():
        drift_rate, decay = condition
        slug = f"drift-{drift_rate}__{CONDITION_SLUGS[decay]}"
        source_directory = job_directories[condition]
        result = condition_results[condition]
        configuration = result["conditions"][0]["configuration"]
        source_result = source_directory / "result.json"
        source_stats = source_directory / "abstention-drift-stats.json"
        source_metadata = source_directory / "metadata.json"
        output_directory = source_root / slug
        output_directory.mkdir(parents=True, exist_ok=True)

        copied_result = output_directory / "result.json"
        copied_stats = output_directory / "abstention-drift-stats.json"
        compressed_metadata = output_directory / "metadata.json.gz"
        shutil.copy2(source_result, copied_result)
        shutil.copy2(source_stats, copied_stats)
        compress_json(source_metadata, compressed_metadata)
        exported_paths.extend(
            [copied_result, copied_stats, compressed_metadata]
        )

        plot_path = plot_root / f"{slug}.png"
        plot_average_results(
            job_id,
            configuration,
            condition_runs[condition],
            plot_path,
        )
        plot_paths[condition] = plot_path
        exported_paths.append(plot_path)

        run_files = sorted(source_directory.glob("run-*.json"))
        legacy_lambda = float(configuration.get("lambda", 1.0))
        manifest_rows.append(
            {
                "drift_rate": int(drift_rate),
                "decay": decay,
                "lambda_rag": float(
                    configuration.get("lambda_rag", legacy_lambda)
                ),
                "lambda_fea": float(
                    configuration.get("lambda_fea", legacy_lambda)
                ),
                "job_id": job_id,
                "started_at": result.get("started_at"),
                "completed_at": result.get("completed_at"),
                "source_job_directory": source_directory.name,
                "source_result_sha256": sha256_file(source_result),
                "included_result": copied_result.relative_to(destination).as_posix(),
                "included_result_sha256": sha256_file(copied_result),
                "included_stats": copied_stats.relative_to(destination).as_posix(),
                "included_stats_sha256": sha256_file(copied_stats),
                "included_metadata_gzip": compressed_metadata.relative_to(
                    destination
                ).as_posix(),
                "included_metadata_gzip_sha256": sha256_file(
                    compressed_metadata
                ),
                "included_plot": plot_path.relative_to(destination).as_posix(),
                "source_run_file_count": len(run_files),
                "source_run_bytes": sum(path.stat().st_size for path in run_files),
                "raw_run_files_included": False,
            }
        )

    legend_path = plot_root / "trajectory-legend.png"
    plot_legend(legend_path)
    combined_path = plot_root / "all-eight-settings.png"
    compose_plot_grid(
        [plot_paths[condition] for condition in PLOT_ORDER],
        legend_path,
        combined_path,
        columns=2,
    )
    exported_paths.extend([legend_path, combined_path])
    return exported_paths, manifest_rows


def export_statistics(output_dir: Path, destination: Path) -> list[Path]:
    condition_rows: dict[tuple[str, str], list[dict]] = {}
    condition_runs: dict[tuple[str, str], list[dict]] = {}
    condition_results: dict[tuple[str, str], dict] = {}
    job_directories: dict[tuple[str, str], Path] = {}
    all_rows = []
    for condition, job_id in PAPER_JOBS.items():
        job_directory = find_job_directory(job_id, output_dir)
        result, runs = load_completed_runs(job_directory, job_id)
        condition_results[condition] = result
        condition_runs[condition] = runs
        job_directories[condition] = job_directory
        rows = [per_run_metrics(run, *condition) for run in runs]
        condition_rows[condition] = rows
        all_rows.extend(rows)

    summary_rows = []
    summarized_fields = [
        *RATE_FIELDS,
        "final_fea",
        "human_only_error_count",
        "final_error_count",
        "stable_error_count",
        "drift_error_count",
        "model_finalized_count",
        "abstain_so_count",
        "abstain_sc_count",
        "abstain_ds_count",
        "sc_accepted_count",
        "sc_acceptance_opportunities",
        "state_so_tickets",
        "state_sc_tickets",
        "state_ds_tickets",
        "assigned_ceo_count",
        "assigned_domain_expert_count",
        "assigned_intern_count",
    ]
    for (drift_rate, decay), rows in condition_rows.items():
        for metric in summarized_fields:
            summary_rows.append(
                {
                    "drift_rate": int(drift_rate),
                    "decay": decay,
                    "metric": metric,
                    **summarize([float(row[metric]) for row in rows]),
                }
            )

    comparisons = paired_comparisons(condition_rows)
    factorial_results = factorial_comparisons(condition_rows)
    baseline_results = baseline_comparisons(condition_rows)
    controller_results = controller_comparisons(condition_rows)
    judge_audit = auxiliary_judge_audit(condition_runs, settings.data_dir)
    state_rows = []
    for (drift_rate, decay), rows in condition_rows.items():
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["final_state"]] = counts.get(row["final_state"], 0) + 1
        state_rows.append(
            {
                "drift_rate": int(drift_rate),
                "decay": decay,
                "n": len(rows),
                "final_state_counts": counts,
            }
        )

    state_csv_rows = []
    for row in state_rows:
        counts = row["final_state_counts"]
        state_csv_rows.append(
            {
                "drift_rate": row["drift_rate"],
                "decay": row["decay"],
                "repetitions": row["n"],
                "silent_observer": counts.get("silent_observer", 0),
                "skeptical_contestator": counts.get(
                    "skeptical_contestator", 0
                ),
                "deferring_surrogate": counts.get("deferring_surrogate", 0),
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    per_run_path = destination / "per-run-results.csv"
    write_csv(per_run_path, all_rows, list(all_rows[0]))

    summary_path = destination / "condition-summary.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0]))

    paired_path = destination / "paired-comparisons.csv"
    paired_csv_rows = [
        {key: value for key, value in row.items() if key != "paired_differences"}
        for row in comparisons
    ]
    write_csv(paired_path, paired_csv_rows, list(paired_csv_rows[0]))

    factorial_path = destination / "factorial-comparisons.csv"
    factorial_csv_rows = [
        {key: value for key, value in row.items() if key != "paired_differences"}
        for row in factorial_results
    ]
    write_csv(
        factorial_path,
        factorial_csv_rows,
        list(factorial_csv_rows[0]),
    )

    baseline_path = destination / "baseline-comparisons.csv"
    baseline_csv_rows = [
        {key: value for key, value in row.items() if key != "paired_differences"}
        for row in baseline_results
    ]
    write_csv(baseline_path, baseline_csv_rows, list(baseline_csv_rows[0]))

    controller_path = destination / "controller-comparisons.csv"
    controller_csv_rows = [
        {key: value for key, value in row.items() if key != "paired_differences"}
        for row in controller_results
    ]
    write_csv(
        controller_path,
        controller_csv_rows,
        list(controller_csv_rows[0]),
    )

    final_states_path = destination / "final-state-frequencies.csv"
    write_csv(
        final_states_path,
        state_csv_rows,
        list(state_csv_rows[0]),
    )

    judge_audit_path = destination / "auxiliary-judge-audit.json"
    judge_audit_path.write_text(
        json.dumps(judge_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source_paths, artifact_manifest_rows = export_source_artifacts(
        destination,
        condition_results,
        condition_runs,
        job_directories,
    )
    artifact_manifest_path = destination / "artifact-manifest.csv"
    write_csv(
        artifact_manifest_path,
        artifact_manifest_rows,
        list(artifact_manifest_rows[0]),
    )

    json_path = destination / "paper-statistics.json"
    json_path.write_text(
        json.dumps(
            {
                "jobs": {
                    f"drift_{drift_rate}_{decay}": job_id
                    for (drift_rate, decay), job_id in PAPER_JOBS.items()
                },
                "definitions": {
                    "condition_sd": "sample standard deviation across repetitions",
                    "condition_ci": (
                        "two-sided 95% Student-t confidence interval for the mean"
                    ),
                    "paired_difference": (
                        "coupled decay minus no decay in percentage points"
                    ),
                    "paired_ci": (
                        "two-sided 95% Student-t confidence interval for the "
                        "matched mean difference"
                    ),
                    "paired_test": (
                        "two-sided exact sign-flip randomization test on the "
                        "matched mean difference"
                    ),
                    "static_rag_defer": (
                        "the model finalizes every non-abstaining proposal and "
                        "the initially assigned human answers on abstention"
                    ),
                    "controller_comparison": (
                        "static RAG-with-defer error minus iRAG final-decision "
                        "error on the same realised retrieval trajectory"
                    ),
                    "multiple_testing": (
                        "Holm adjustment across four primary metrics within each "
                        "paired contrast; baseline and controller comparisons are "
                        "adjusted across the eight paper conditions"
                    ),
                },
                "condition_summaries": summary_rows,
                "paired_comparisons": comparisons,
                "factorial_comparisons": factorial_results,
                "baseline_comparisons": baseline_results,
                "controller_comparisons": controller_results,
                "auxiliary_judge_audit": judge_audit,
                "final_states": state_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        per_run_path,
        summary_path,
        paired_path,
        factorial_path,
        baseline_path,
        controller_path,
        final_states_path,
        judge_audit_path,
        artifact_manifest_path,
        json_path,
        *source_paths,
    ]


def main() -> int:
    args = parse_args()
    for path in export_statistics(args.output_dir, args.destination):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
