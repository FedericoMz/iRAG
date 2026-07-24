from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse

from irag import __version__
from irag.client import BedrockClient, BaseModelClient, OllamaClient, OpenRouterClient
from irag.core.config import settings
from irag.core.models import (
    AcceptanceRegime,
    AssignmentStrategy,
    BundledExperimentRequest,
    ExperimentCondition,
    ExperimentCreated,
    ExperimentRequest,
    ExperimentStatus,
    ExpertSelection,
    JobStatus,
    ModelSettings,
    ModelProvider,
    ParallelRunRequest,
    Profile,
    Quarter,
    QuarterBatch,
    RunAcceptance,
)
from irag.data.dataset import SalesXDataset
from irag.data.store import ExperimentStore
from irag.engine.experiment import ExperimentRunner
from irag.tools.logger import logger


app = FastAPI(
    title="SalesX Incremental RAG Experiments",
    description=(
        "Experiment-specific API for the incremental deliberative RAG evaluation. "
        "It is intentionally bound to the SalesX quarterly corpus, post-Q4 "
        "abstention challenge, and their precomputed embeddings."
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
            f"__extra-{str(request.include_extra).lower()}"
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
    "/v1/runs/{experiment_id}/resume",
    response_model=ExperimentCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume a checkpointed parallel run",
)
def resume_parallel_run(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    checkpoint_interval: Annotated[int, Query(ge=1, le=500)] = 50,
) -> ExperimentCreated:
    job = store.get(experiment_id)
    if job is not None and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(status_code=409, detail="Experiment is already active")
    if job is not None and job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Experiment is already complete")

    try:
        metadata = (
            store.read_metadata(experiment_id)
            if job is not None
            else store.read_saved_metadata(experiment_id)
        )
        request = build_resume_request(metadata)
        if len(request.conditions) != 1 or request.conditions[0].repetitions != 1:
            raise ValueError(
                "Resume currently supports parallel runs with one repetition"
            )
        if job is None:
            started_at = datetime.fromisoformat(metadata["started_at"])
            job = store.recover(experiment_id, request, started_at)
        if store.run_path(experiment_id, 1).exists():
            raise ValueError("The saved repetition is already finalized")
        tickets = store.read_partial_tickets(experiment_id, 1)
        total_tickets = sum(len(batch.records) for batch in request.quarters)
        if len(tickets) > total_tickets:
            raise ValueError("Checkpoint contains more tickets than the experiment")
        for position, ticket in enumerate(tickets, start=1):
            if ticket.get("global_position") != position:
                raise ValueError(
                    f"Checkpoint is not contiguous at saved ticket {position}"
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    job = store.queue_resume(experiment_id)
    background_tasks.add_task(
        run_parallel_experiment,
        experiment_id,
        request,
        checkpoint_interval,
        {1: tickets},
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
    elif request.expert == ExpertSelection.INFORMED_MIXTURE:
        assignment = AssignmentStrategy.INFORMED
        profile = None
    else:
        assignment = AssignmentStrategy.CEO_BOOTSTRAPPED_INFORMED
        profile = None

    acceptance = {
        RunAcceptance.ALWAYS_REFUSE: AcceptanceRegime.NEVER,
        RunAcceptance.ALWAYS_ACCEPT: AcceptanceRegime.ALWAYS,
        RunAcceptance.RANDOMIZE: AcceptanceRegime.STOCHASTIC,
        RunAcceptance.GOLD_SIMILARITY: AcceptanceRegime.GOLD_SIMILARITY,
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
        quarters=dataset.load_all_quarters(include_extra=request.include_extra),
        conditions=[condition],
        models={
            "provider": request.provider,
            "generation_model": request.generation_model,
            "auxiliary_model": request.auxiliary_model,
            "ollama_base_url": request.ollama_base_url,
            "bedrock_region": request.bedrock_region,
            "timeout": request.timeout,
            "retries": request.retries,
        },
    )


def build_resume_request(metadata: dict) -> ExperimentRequest:
    models = metadata["models"]
    generation = models["generation"]
    auxiliary = models["auxiliary"]
    include_extra = any(
        record.get("quarter") == Quarter.EXTRA.value
        for record in metadata.get("records", {}).values()
    )
    return ExperimentRequest(
        name=metadata["name"],
        quarters=dataset.load_all_quarters(include_extra=include_extra),
        conditions=[ExperimentCondition.model_validate(metadata["configuration"])],
        models=ModelSettings(
            provider=models["provider"],
            generation_model=generation["name"],
            auxiliary_model=auxiliary["name"],
            bedrock_region=generation.get("region") or auxiliary.get("region"),
            retries=generation.get("max_attempts"),
        ),
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
    resume_tickets: dict[int, list[dict]] | None = None,
) -> None:
    logger.info(f"Starting parallel experiment {experiment_id}: {request.name}")
    store.start(experiment_id)
    condition = request.conditions[0]
    configuration = condition.model_dump(mode="json", by_alias=True)

    def write_metadata(result: dict) -> None:
        if resume_tickets is not None:
            existing = store.read_metadata(experiment_id)
            resume_event = {
                "resumed_at": result["started_at"],
                "checkpoint_tickets": {
                    str(repetition): len(tickets)
                    for repetition, tickets in resume_tickets.items()
                },
            }
            result["started_at"] = existing.get("started_at", result["started_at"])
            result["resume_events"] = [
                *existing.get("resume_events", []),
                resume_event,
            ]
            existing["resume_events"] = result["resume_events"]
            store.write_metadata(experiment_id, existing)
            return
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
            resume_tickets=resume_tickets,
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
            "MODEL_PROVIDER must be 'ollama', 'openrouter', or 'bedrock'"
        ) from exc

    timeout = models.timeout or settings.model_timeout
    retries = models.retries or (
        settings.bedrock_retries
        if provider == ModelProvider.BEDROCK
        else settings.model_retries
    )
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
    if provider == ModelProvider.OPENROUTER:
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            generation_model=(
                models.generation_model or settings.openrouter_generation_model
            ),
            auxiliary_model=(
                models.auxiliary_model or settings.openrouter_auxiliary_model
            ),
            timeout=timeout,
            retries=retries,
            http_referer=settings.openrouter_http_referer,
            app_title=settings.openrouter_app_title,
        )
    return BedrockClient(
        region=models.bedrock_region or settings.bedrock_region,
        profile=settings.bedrock_profile,
        generation_model=(
            models.generation_model or settings.bedrock_generation_model
        ),
        auxiliary_model=(models.auxiliary_model or settings.bedrock_auxiliary_model),
        timeout=timeout,
        retries=retries,
        throttle_retries=settings.bedrock_throttle_retries,
        throttle_max_delay=settings.bedrock_throttle_max_delay,
        response_retries=settings.bedrock_response_retries,
        response_max_delay=settings.bedrock_response_max_delay,
        decision_response_retries=settings.bedrock_decision_response_retries,
    )
