from pathlib import Path

from irag.core.models import Quarter
from irag.data.dataset import SalesXDataset
from irag.tools.run_sample_experiment import sample_quarter


ROOT = Path(__file__).resolve().parents[1]


def load_dataset(relative_path: str) -> SalesXDataset:
    return SalesXDataset(
        ROOT / relative_path,
        "qwen3-embedding:4b",
    )


def test_40_percent_variant_reuses_embeddings_for_identical_questions():
    base = load_dataset("experiment data/drift_10")
    variant = load_dataset("experiment data/drift_40")
    base_batch = base.load_quarter(Quarter.Q2)
    variant_batch = variant.load_quarter(Quarter.Q2)

    assert {
        record.id: record.question for record in variant_batch.records
    } == {
        record.id: record.question for record in base_batch.records
    }
    assert sum(
        record.is_changed_answer_near_duplicate
        for record in variant_batch.records
    ) == 200

    variant.validate_batches([variant_batch])
    assert variant.vector(variant_batch.records[0].id).shape == (2560,)


def test_sample_preserves_each_dataset_drift_rate():
    base_sample = sample_quarter(
        load_dataset("experiment data/drift_10"),
        Quarter.Q2,
        tickets=20,
        seed=17,
    )
    variant_sample = sample_quarter(
        load_dataset("experiment data/drift_40"),
        Quarter.Q2,
        tickets=20,
        seed=17,
    )

    assert sum(
        record.is_changed_answer_near_duplicate
        for record in base_sample.records
    ) == 2
    assert sum(
        record.is_changed_answer_near_duplicate
        for record in variant_sample.records
    ) == 8


def test_variants_load_the_same_shared_q1_and_extra_records():
    base = load_dataset("experiment data/drift_10")
    variant = load_dataset("experiment data/drift_40")

    for quarter in (Quarter.Q1, Quarter.EXTRA):
        assert base.load_quarter(quarter) == variant.load_quarter(quarter)

    assert base.manifest()["quarters"]["Q1"]["questions"] == (
        "../shared/Q1_qa.json"
    )
    assert variant.manifest()["quarters"]["Q1"]["questions"] == (
        "../shared/Q1_qa.json"
    )
