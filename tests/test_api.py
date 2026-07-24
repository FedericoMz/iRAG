import json
from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient

import irag.api as api
from irag.client import OpenRouterClient
from irag.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    ExperimentCondition,
    ExperimentRequest,
    JobStatus,
    ModelSettings,
    ParallelRunRequest,
    Profile,
    Quarter,
    QuarterBatch,
)
from irag.store import ExperimentStore
from tests.test_experiment import FakeClient, FakeDataset, make_ticket


def test_health_endpoint():
    response = TestClient(api.app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_bundled_dataset_exposes_post_q4_abstention_split():
    batch = api.dataset.load_quarter(Quarter.EXTRA)

    assert batch.quarter == Quarter.EXTRA
    assert len(batch.records) == 50
    assert all(record.requires_model_abstention for record in batch.records)
    assert api.dataset.vector(batch.records[0].id).shape == (2560,)


def test_experiment_endpoint_accepts_complete_ticket_metadata(monkeypatch):
    monkeypatch.setattr(api, "run_experiment", lambda *_: None)
    record = api.dataset.load_quarter(Quarter.Q1).records[0]
    payload = {
        "name": "contract test",
        "models": {"provider": "ollama"},
        "quarters": [
            {
                "quarter": "Q1",
                "records": [record.model_dump(mode="json")],
            }
        ],
        "conditions": [
            {
                "name": "baseline",
                "assignment_strategy": "single_profile",
                "single_profile": "ceo",
                "acceptance_regime": "never_accept",
                "repetitions": 1,
                "system_enabled": False,
            }
        ],
    }

    response = TestClient(api.app).post("/v1/experiments", json=payload)

    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_model_factory_selects_openrouter_without_putting_key_in_request(monkeypatch):
    monkeypatch.setattr(
        api,
        "settings",
        replace(api.settings, openrouter_api_key="environment-key"),
    )
    request = SimpleNamespace(
        models=ModelSettings(
            provider="openrouter",
            generation_model="vendor/generation",
            auxiliary_model="vendor/judge",
        )
    )

    client = api.make_model_client(request)

    assert isinstance(client, OpenRouterClient)
    assert client.api_key == "environment-key"


def test_model_factory_selects_bedrock_without_api_credentials(monkeypatch):
    captured = {}

    def fake_bedrock_client(**arguments):
        captured.update(arguments)
        return SimpleNamespace(provider="bedrock")

    monkeypatch.setattr(api, "BedrockClient", fake_bedrock_client)
    request = SimpleNamespace(
        models=ModelSettings(
            provider="bedrock",
            generation_model="eu.vendor/generation",
            auxiliary_model="eu.vendor/judge",
            bedrock_region="eu-west-1",
        )
    )

    client = api.make_model_client(request)

    assert client.provider == "bedrock"
    assert captured["region"] == "eu-west-1"
    assert captured["profile"] == api.settings.bedrock_profile
    assert captured["generation_model"] == "eu.vendor/generation"
    assert captured["auxiliary_model"] == "eu.vendor/judge"


def test_parallel_run_endpoint_uses_dropdown_values_and_parameter_folder(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(api, "store", ExperimentStore(tmp_path))
    submitted = []
    monkeypatch.setattr(
        api,
        "run_parallel_experiment",
        lambda _, request, *args: submitted.append(request),
    )

    response = TestClient(api.app).post(
        "/v1/runs",
        params={
            "expert": "informed_mixture",
            "acceptance": "randomize",
            "repetitions": 2,
            "decay": 0.99,
            "provider": "openrouter",
            "generation_model": "vendor/decision-model",
            "auxiliary_model": "vendor/judge-model",
            "timeout": 120,
            "retries": 2,
        },
    )

    assert response.status_code == 202
    experiment_id = response.json()["experiment_id"]
    status_response = TestClient(api.app).get(f"/v1/experiments/{experiment_id}")
    output_directory = status_response.json()["output_directory"]
    assert (
        "expert-informed_mixture__acceptance-randomize"
        "__repetitions-2__decay-0.99__extra-false__job-"
    ) in output_directory
    assert submitted[0].models.provider.value == "openrouter"
    assert submitted[0].models.generation_model == "vendor/decision-model"
    assert submitted[0].models.auxiliary_model == "vendor/judge-model"
    assert submitted[0].models.timeout == 120
    assert submitted[0].models.retries == 2


def test_parallel_run_schema_exposes_expert_and_acceptance_enums():
    schemas = api.app.openapi()["components"]["schemas"]

    assert schemas["ExpertSelection"]["enum"] == [
        "ceo",
        "domain_expert",
        "intern",
        "random_mixture",
        "informed_mixture",
        "ceo_bootstrapped_informed_mixture",
    ]
    assert schemas["RunAcceptance"]["enum"] == [
        "always_refuse",
        "always_accept",
        "randomize",
        "gold_similarity",
    ]
    assert schemas["ModelProvider"]["enum"] == [
        "ollama",
        "openrouter",
        "bedrock",
    ]
    parameters = {
        parameter["name"]: parameter
        for parameter in api.app.openapi()["paths"]["/v1/runs"]["post"]["parameters"]
    }
    assert parameters["provider"]["schema"]["anyOf"][0]["$ref"].endswith(
        "/ModelProvider"
    )
    assert "generation_model" in parameters
    assert "auxiliary_model" in parameters
    assert "bedrock_region" in parameters
    assert parameters["checkpoint_interval"]["schema"]["default"] == 50
    assert parameters["include_extra"]["schema"]["default"] is False


def test_openapi_does_not_expose_obsolete_paper_suite_endpoints():
    paths = api.app.openapi()["paths"]

    assert "/v1/experiments/paper-suite" not in paths
    assert "/v1/experiments/paper-suite/bundled" not in paths


def test_parallel_request_supports_ceo_bootstrapped_informed_mixture():
    request = ParallelRunRequest(
        expert="ceo_bootstrapped_informed_mixture",
        acceptance="gold_similarity",
        repetitions=1,
    )

    built = api.build_parallel_request(request)
    condition = built.conditions[0]

    assert (
        condition.assignment_strategy
        == AssignmentStrategy.CEO_BOOTSTRAPPED_INFORMED
    )
    assert condition.alpha == 0.7
    assert condition.gamma == 0.8
    assert condition.quarterly_ceo_tickets == 100
    assert condition.acceptance_regime == AcceptanceRegime.GOLD_SIMILARITY
    assert [batch.quarter for batch in built.quarters] == [
        Quarter.Q1,
        Quarter.Q2,
        Quarter.Q3,
        Quarter.Q4,
    ]


def test_parallel_request_includes_extra_only_when_requested():
    request = ParallelRunRequest(
        expert="informed_mixture",
        acceptance="gold_similarity",
        repetitions=1,
        include_extra=True,
    )

    built = api.build_parallel_request(request)

    assert [batch.quarter for batch in built.quarters] == [
        Quarter.Q1,
        Quarter.Q2,
        Quarter.Q3,
        Quarter.Q4,
        Quarter.EXTRA,
    ]


def test_resume_reconstructs_extra_scope_from_saved_records():
    condition = api.build_parallel_request(
        ParallelRunRequest(
            expert="informed_mixture",
            acceptance="gold_similarity",
            repetitions=1,
        )
    ).conditions[0]
    metadata = {
        "name": "resume scope",
        "configuration": condition.model_dump(mode="json", by_alias=True),
        "models": {
            "provider": "bedrock",
            "generation": {"name": "generation"},
            "auxiliary": {"name": "auxiliary"},
        },
        "records": {"SX-Q4-BIL-001": {"quarter": "Q4"}},
    }

    nominal = api.build_resume_request(metadata)
    metadata["records"]["SX-EXTRA-001"] = {"quarter": "Extra"}
    with_extra = api.build_resume_request(metadata)

    assert nominal.quarters[-1].quarter == Quarter.Q4
    assert with_extra.quarters[-1].quarter == Quarter.EXTRA


def test_parallel_background_job_persists_each_repetition(monkeypatch, tmp_path):
    experiment_store = ExperimentStore(tmp_path)
    monkeypatch.setattr(api, "store", experiment_store)
    monkeypatch.setattr(api, "dataset", FakeDataset())
    monkeypatch.setattr(api, "make_model_client", lambda _: FakeClient())
    request = ExperimentRequest(
        name="parallel persistence",
        quarters=[
            QuarterBatch(
                quarter=Quarter.Q1,
                records=[make_ticket(number) for number in range(1, 4)],
            )
        ],
        conditions=[
            ExperimentCondition(
                name="parallel-persistence",
                assignment_strategy=AssignmentStrategy.SINGLE,
                single_profile=Profile.CEO,
                acceptance_regime=AcceptanceRegime.NEVER,
                repetitions=2,
            )
        ],
    )
    job = experiment_store.create(request, "parallel-persistence")

    api.run_parallel_experiment(job.experiment_id, request)

    completed = experiment_store.get(job.experiment_id)
    assert completed.status == JobStatus.COMPLETED
    assert experiment_store.metadata_path(job.experiment_id).is_file()
    assert experiment_store.result_path(job.experiment_id).is_file()
    assert [path.name for path in experiment_store.list_runs(job.experiment_id)] == [
        "run-001.json",
        "run-002.json",
    ]
    first_run = json.loads(
        experiment_store.run_path(job.experiment_id, 1).read_text(encoding="utf-8")
    )
    assert len(first_run["tickets"]) == 3
    assert not experiment_store.partial_run_path(job.experiment_id, 1).exists()


def test_resume_endpoint_recovers_job_after_process_restart(monkeypatch, tmp_path):
    request = ExperimentRequest(
        name="resumable run",
        quarters=[QuarterBatch(quarter=Quarter.Q1, records=[make_ticket(1)])],
        conditions=[
            ExperimentCondition(
                name="resumable",
                assignment_strategy=AssignmentStrategy.SINGLE,
                single_profile=Profile.CEO,
                acceptance_regime=AcceptanceRegime.NEVER,
                repetitions=1,
            )
        ],
    )
    old_store = ExperimentStore(tmp_path)
    old_job = old_store.create(request, "resumable")
    old_store.write_metadata(
        old_job.experiment_id,
        {
            "name": request.name,
            "started_at": old_job.created_at.isoformat(),
        },
    )
    old_store.append_run_tickets(
        old_job.experiment_id,
        1,
        [{"global_position": 1}],
    )

    restarted_store = ExperimentStore(tmp_path)
    submitted = []
    monkeypatch.setattr(api, "store", restarted_store)
    monkeypatch.setattr(api, "build_resume_request", lambda _: request)
    monkeypatch.setattr(
        api,
        "run_parallel_experiment",
        lambda *arguments: submitted.append(arguments),
    )

    response = TestClient(api.app).post(
        f"/v1/runs/{old_job.experiment_id}/resume"
    )

    assert response.status_code == 202
    assert response.json()["experiment_id"] == old_job.experiment_id
    assert response.json()["status"] == "queued"
    assert submitted[0][0] == old_job.experiment_id
    assert submitted[0][3] == {1: [{"global_position": 1}]}
    recovered = restarted_store.get(old_job.experiment_id)
    assert recovered is not None
    assert recovered.output_directory == old_job.output_directory
