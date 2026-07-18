from irag.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    ExperimentCondition,
    ExperimentRequest,
    Profile,
    Quarter,
    QuarterBatch,
)
from irag.store import ExperimentStore

from .test_experiment import make_ticket


def test_store_writes_metadata_summary_and_numbered_runs(tmp_path):
    request = ExperimentRequest(
        name="store test",
        quarters=[QuarterBatch(quarter=Quarter.Q1, records=[make_ticket(1)])],
        conditions=[
            ExperimentCondition(
                name="condition",
                assignment_strategy=AssignmentStrategy.SINGLE,
                single_profile=Profile.CEO,
                acceptance_regime=AcceptanceRegime.NEVER,
                repetitions=2,
            )
        ],
    )
    store = ExperimentStore(tmp_path)
    job = store.create(request, "expert-ceo__acceptance-always_refuse")

    store.write_metadata(job.experiment_id, {"kind": "metadata"})
    store.append_run_tickets(job.experiment_id, 2, [{"ticket_id": "ticket-2"}])
    store.append_run_tickets(job.experiment_id, 1, [{"ticket_id": "ticket-1"}])
    store.finalize_run(job.experiment_id, 2, {"repetition": 2})
    store.finalize_run(job.experiment_id, 1, {"repetition": 1})
    store.complete(job.experiment_id, {"kind": "summary"})

    assert store.metadata_path(job.experiment_id).is_file()
    assert store.result_path(job.experiment_id).is_file()
    assert [path.name for path in store.list_runs(job.experiment_id)] == [
        "run-001.json",
        "run-002.json",
    ]
    assert not store.partial_run_path(job.experiment_id, 1).exists()
