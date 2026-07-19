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
        "__repetitions-2__decay-0.99__job-"
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
    ]
    assert schemas["RunAcceptance"]["enum"] == [
        "always_refuse",
        "always_accept",
        "randomize",
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
