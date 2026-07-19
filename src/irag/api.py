from __future__ import annotations

from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse

from irag import __version__
from irag.client import BaseModelClient, OllamaClient, OpenRouterClient
from irag.config import settings
from irag.dataset import SalesXDataset
from irag.experiment import ExperimentRunner, build_paper_request
from irag.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    BundledExperimentRequest,
    BundledPaperSuiteRequest,
    ExperimentCondition,
    ExperimentCreated,
    ExperimentRequest,
    ExperimentStatus,
    ExpertSelection,
    JobStatus,
    ModelProvider,
    ParallelRunRequest,
    PaperSuiteRequest,
    Profile,
    Quarter,
    QuarterBatch,
    RunAcceptance,
)
from irag.store import ExperimentStore
from irag.tools.logger import logger


app = FastAPI(
    title="SalesX Incremental RAG Experiments",
    description=(
        "Experiment-specific API for the incremental deliberative RAG evaluation. "
        "It is intentionally bound to the SalesX quarterly corpus and its precomputed embeddings."
    ),
    version=__version__,
)

dataset = SalesXDataset(settings.data_dir, settings.embedding_model)
store = ExperimentStore(settings.output_dir)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/dataset/quarters/{quarter}", response_model=QuarterBatch)
def get_quarter(quarter: Quarter) -> QuarterBatch:
    try:
        return dataset.load_quarter(quarter)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/v1/experiments",
    response_model=ExperimentCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentCreated:
    return enqueue(request, background_tasks)


@app.post(
    "/v1/runs",
    response_model=ExperimentCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run one configuration with repetitions in parallel",
)
def create_parallel_run(
    background_tasks: BackgroundTasks,
    request: Annotated[ParallelRunRequest, Query()],
) -> ExperimentCreated:
    experiment_request = build_parallel_request(request)
    job = store.create(
        experiment_request,
        folder_label=(
            f"expert-{request.expert.value}"
            f"__acceptance-{request.acceptance.value}"
            f"__repetitions-{request.repetitions}"
            f"__decay-{request.decay:.8g}"
        ),
    )
    background_tasks.add_task(
        run_parallel_experiment,
        job.experiment_id,
        experiment_request,
        request.checkpoint_interval,
    )
    return created_response(job)


@app.post(
    "/v1/experiments/bundled",
    response_model=ExperimentCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_bundled_experiment(
    request: BundledExperimentRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentCreated:
    experiment_request = ExperimentRequest(
        name=request.name,
        quarters=dataset.load_all_quarters(),
        conditions=request.conditions,
        models=request.models,
    )
    return enqueue(experiment_request, background_tasks)


@app.post(
    "/v1/experiments/paper-suite",
    response_model=ExperimentCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_paper_suite(
    request: PaperSuiteRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentCreated:
    return enqueue(build_paper_request(request), background_tasks)


@app.post(
    "/v1/experiments/paper-suite/bundled",
    response_model=ExperimentCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_bundled_paper_suite(
    request: BundledPaperSuiteRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentCreated:
    paper_request = PaperSuiteRequest(
        name=request.name,
        quarters=dataset.load_all_quarters(),
        repetitions=request.repetitions,
        seed=request.seed,
        domain_expert_category=request.domain_expert_category,
        models=request.models,
        include_baselines=request.include_baselines,
        include_decay_ablation=request.include_decay_ablation,
    )
    return enqueue(build_paper_request(paper_request), background_tasks)


@app.get("/v1/experiments/{experiment_id}", response_model=ExperimentStatus)
def get_experiment(experiment_id: str) -> ExperimentStatus:
    job = store.get(experiment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return job


@app.get("/v1/experiments/{experiment_id}/result")
def get_experiment_result(experiment_id: str) -> FileResponse:
    job = store.get(experiment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment result is not ready; current status is {job.status.value}",
        )
    path = store.result_path(experiment_id)
    if not path.exists():
        raise HTTPException(status_code=500, detail="Experiment result file is missing")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{experiment_id}.json",
    )


@app.get("/v1/experiments/{experiment_id}/runs")
def list_experiment_runs(experiment_id: str) -> dict:
    job = store.get(experiment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "experiment_id": experiment_id,
        "status": job.status.value,
        "runs": [path.name for path in store.list_runs(experiment_id)],
    }


@app.get("/v1/experiments/{experiment_id}/metadata")
def get_experiment_metadata(experiment_id: str) -> FileResponse:
    job = store.get(experiment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    path = store.metadata_path(experiment_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment metadata is not available")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/v1/experiments/{experiment_id}/runs/{repetition}")
def get_experiment_run(experiment_id: str, repetition: int) -> FileResponse:
    job = store.get(experiment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    path = store.run_path(experiment_id, repetition)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run output is not available")
    return FileResponse(path, media_type="application/json", filename=path.name)


def enqueue(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentCreated:
    job = store.create(request)
    background_tasks.add_task(run_experiment, job.experiment_id, request)
    return created_response(job)


def created_response(job: ExperimentStatus) -> ExperimentCreated:
    return ExperimentCreated(
        experiment_id=job.experiment_id,
        status=job.status,
        status_url=f"/v1/experiments/{job.experiment_id}",
        result_url=f"/v1/experiments/{job.experiment_id}/result",
    )


def build_parallel_request(request: ParallelRunRequest) -> ExperimentRequest:
    if request.expert == ExpertSelection.CEO:
        assignment = AssignmentStrategy.SINGLE
        profile = Profile.CEO
    elif request.expert == ExpertSelection.DOMAIN_EXPERT:
        assignment = AssignmentStrategy.SINGLE
        profile = Profile.DOMAIN_EXPERT
    elif request.expert == ExpertSelection.INTERN:
        assignment = AssignmentStrategy.SINGLE
        profile = Profile.INTERN
    elif request.expert == ExpertSelection.RANDOM_MIXTURE:
        assignment = AssignmentStrategy.RANDOM
        profile = None
    else:
        assignment = AssignmentStrategy.INFORMED
        profile = None

    acceptance = {
        RunAcceptance.ALWAYS_REFUSE: AcceptanceRegime.NEVER,
        RunAcceptance.ALWAYS_ACCEPT: AcceptanceRegime.ALWAYS,
        RunAcceptance.RANDOMIZE: AcceptanceRegime.STOCHASTIC,
    }[request.acceptance]
    condition = ExperimentCondition(
        name=f"{request.expert.value}__{request.acceptance.value}",
        assignment_strategy=assignment,
        single_profile=profile,
        acceptance_regime=acceptance,
        domain_expert_category=request.domain_expert_category,
        repetitions=request.repetitions,
        seed=request.seed,
        decay=request.decay,
    )
    return ExperimentRequest(
        name=f"{request.expert.value} / {request.acceptance.value}",
        quarters=dataset.load_all_quarters(),
        conditions=[condition],
        models={
            "provider": request.provider,
            "generation_model": request.generation_model,
            "auxiliary_model": request.auxiliary_model,
            "ollama_base_url": request.ollama_base_url,
            "timeout": request.timeout,
            "retries": request.retries,
        },
    )


def run_experiment(experiment_id: str, request: ExperimentRequest) -> None:
    logger.info(f"Starting experiment {experiment_id}: {request.name}")
    store.start(experiment_id)
    try:
        client = make_model_client(request)
        runner = ExperimentRunner(dataset, client)
        result = runner.run(
            experiment_id,
            request,
            progress=lambda condition, completed: store.progress(
                experiment_id, condition, completed
            ),
        )
        path = store.complete(experiment_id, result)
        logger.info(f"Experiment {experiment_id} completed: {path}")
    except Exception as exc:
        logger.exception(f"Experiment {experiment_id} failed")
        store.fail(experiment_id, str(exc))


def run_parallel_experiment(
    experiment_id: str,
    request: ExperimentRequest,
    checkpoint_interval: int = 50,
) -> None:
    logger.info(f"Starting parallel experiment {experiment_id}: {request.name}")
    store.start(experiment_id)
    condition = request.conditions[0]
    configuration = condition.model_dump(mode="json", by_alias=True)

    def write_metadata(result: dict) -> None:
        metadata = {
            key: value
            for key, value in result.items()
            if key not in {"conditions", "completed_at"}
        }
        metadata["configuration"] = configuration
        store.write_metadata(experiment_id, metadata)

    def write_repetition(run_result: dict) -> str:
        payload = {
            "experiment_id": experiment_id,
            "metadata_file": "metadata.json",
            "configuration": configuration,
            **run_result,
        }
        return store.finalize_run(
            experiment_id,
            run_result["repetition"],
            payload,
        ).name

    def write_ticket_batch(repetition: int, tickets: list[dict]) -> None:
        store.append_run_tickets(experiment_id, repetition, tickets)

    try:
        client = make_model_client(request)
        runner = ExperimentRunner(dataset, client)
        result = runner.run_parallel(
            experiment_id,
            request,
            progress=lambda condition_name, completed: store.progress(
                experiment_id, condition_name, completed
            ),
            on_metadata=write_metadata,
            on_ticket_batch=write_ticket_batch,
            ticket_batch_size=checkpoint_interval,
            on_repetition=write_repetition,
        )
        summary = {
            key: value
            for key, value in result.items()
            if key != "records"
        }
        summary["metadata_file"] = "metadata.json"
        path = store.complete(experiment_id, summary)
        logger.info(f"Parallel experiment {experiment_id} completed: {path}")
    except Exception as exc:
        logger.exception(f"Parallel experiment {experiment_id} failed")
        store.fail(experiment_id, str(exc))


def make_model_client(request: ExperimentRequest) -> BaseModelClient:
    models = request.models
    try:
        provider = models.provider or ModelProvider(settings.model_provider.lower())
    except ValueError as exc:
        raise ValueError(
            "MODEL_PROVIDER must be either 'ollama' or 'openrouter'"
        ) from exc

    timeout = models.timeout or settings.model_timeout
    retries = models.retries or settings.model_retries
    if provider == ModelProvider.OLLAMA:
        return OllamaClient(
            base_url=models.ollama_base_url or settings.ollama_base_url,
            generation_model=(
                models.generation_model or settings.ollama_generation_model
            ),
            auxiliary_model=(models.auxiliary_model or settings.ollama_auxiliary_model),
            timeout=timeout,
            retries=retries,
        )
    return OpenRouterClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        generation_model=(
            models.generation_model or settings.openrouter_generation_model
        ),
        auxiliary_model=(models.auxiliary_model or settings.openrouter_auxiliary_model),
        timeout=timeout,
        retries=retries,
        http_referer=settings.openrouter_http_referer,
        app_title=settings.openrouter_app_title,
    )
