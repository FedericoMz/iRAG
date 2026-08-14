from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from irag.core.models import ExperimentRequest, ExperimentStatus, JobStatus


class ExperimentStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, ExperimentStatus] = {}
        self._directories: dict[str, Path] = {}
        self._lock = threading.Lock()

    def create(
        self,
        request: ExperimentRequest,
        folder_label: str = "experiment",
    ) -> ExperimentStatus:
        return self.create_job(
            name=request.name,
            total_repetitions=sum(
                condition.repetitions for condition in request.conditions
            ),
            folder_label=folder_label,
        )

    def create_job(
        self,
        name: str,
        total_repetitions: int,
        folder_label: str,
    ) -> ExperimentStatus:
        """Create a persisted asynchronous job without an iRAG request schema."""
        if total_repetitions < 1:
            raise ValueError("total_repetitions must be at least 1")
        experiment_id = uuid4().hex
        safe_label = re.sub(r"[^a-zA-Z0-9._=-]+", "-", folder_label).strip("-")
        directory = self.output_dir / f"{safe_label}__job-{experiment_id}"
        directory.mkdir(parents=True, exist_ok=False)
        status = ExperimentStatus(
            experiment_id=experiment_id,
            name=name,
            status=JobStatus.QUEUED,
            created_at=datetime.now(UTC),
            total_repetitions=total_repetitions,
            output_directory=str(directory.resolve()),
        )
        with self._lock:
            self._jobs[experiment_id] = status
            self._directories[experiment_id] = directory
        return status

    def get(self, experiment_id: str) -> ExperimentStatus | None:
        with self._lock:
            status = self._jobs.get(experiment_id)
            return status.model_copy(deep=True) if status else None

    def recover(
        self,
        experiment_id: str,
        request: ExperimentRequest,
        created_at: datetime,
    ) -> ExperimentStatus:
        directory = self.saved_directory(experiment_id)
        status = ExperimentStatus(
            experiment_id=experiment_id,
            name=request.name,
            status=JobStatus.FAILED,
            created_at=created_at,
            completed_at=datetime.now(UTC),
            total_repetitions=sum(
                condition.repetitions for condition in request.conditions
            ),
            output_directory=str(directory.resolve()),
            error="Recovered from a previous process",
        )
        with self._lock:
            self._jobs[experiment_id] = status
            self._directories[experiment_id] = directory
        return status.model_copy(deep=True)

    def saved_directory(self, experiment_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", experiment_id):
            raise ValueError("Invalid experiment ID")
        matches = list(self.output_dir.glob(f"*__job-{experiment_id}"))
        if len(matches) != 1 or not matches[0].is_dir():
            raise ValueError(f"Could not find saved experiment {experiment_id}")
        return matches[0]

    def read_saved_metadata(self, experiment_id: str) -> dict:
        source = self.saved_directory(experiment_id) / "metadata.json"
        if not source.is_file():
            raise ValueError("Saved experiment metadata is missing")
        return json.loads(source.read_text(encoding="utf-8"))

    def read_saved_run(self, experiment_id: str, repetition: int) -> dict:
        source = (
            self.saved_directory(experiment_id)
            / f"run-{repetition:03d}.json"
        )
        if not source.is_file():
            raise ValueError(f"Finalized repetition {repetition} is missing")
        result = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"Finalized repetition {repetition} is invalid")
        return result

    def queue_resume(self, experiment_id: str) -> ExperimentStatus:
        self._update(
            experiment_id,
            status=JobStatus.QUEUED,
            started_at=None,
            completed_at=None,
            current_condition=None,
            completed_repetitions=0,
            error=None,
        )
        status = self.get(experiment_id)
        assert status is not None
        return status

    def start(self, experiment_id: str) -> None:
        self._update(
            experiment_id,
            status=JobStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

    def progress(self, experiment_id: str, condition: str, completed: int) -> None:
        self._update(
            experiment_id,
            current_condition=condition,
            completed_repetitions=completed,
        )

    def complete(self, experiment_id: str, result: dict) -> Path:
        destination = self.result_path(experiment_id)
        self._write_json(destination, result)
        status = self.get(experiment_id)
        self._update(
            experiment_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            current_condition=None,
            completed_repetitions=(status.total_repetitions if status else 0),
        )
        return destination

    def fail(self, experiment_id: str, error: str) -> None:
        self._update(
            experiment_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now(UTC),
            error=error,
        )

    def result_path(self, experiment_id: str) -> Path:
        return self.directory(experiment_id) / "result.json"

    def metadata_path(self, experiment_id: str) -> Path:
        return self.directory(experiment_id) / "metadata.json"

    def run_path(self, experiment_id: str, repetition: int) -> Path:
        return self.directory(experiment_id) / f"run-{repetition:03d}.json"

    def partial_run_path(self, experiment_id: str, repetition: int) -> Path:
        return self.directory(experiment_id) / f"run-{repetition:03d}.partial.jsonl"

    def write_metadata(self, experiment_id: str, metadata: dict) -> Path:
        destination = self.metadata_path(experiment_id)
        self._write_json(destination, metadata)
        return destination

    def read_metadata(self, experiment_id: str) -> dict:
        return json.loads(self.metadata_path(experiment_id).read_text(encoding="utf-8"))

    def read_partial_tickets(
        self,
        experiment_id: str,
        repetition: int,
    ) -> list[dict]:
        source = self.partial_run_path(experiment_id, repetition)
        if not source.exists():
            return []
        tickets = []
        with source.open(encoding="utf-8") as ticket_input:
            for line_number, line in enumerate(ticket_input, start=1):
                if not line.strip():
                    continue
                try:
                    ticket = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid checkpoint JSON on line {line_number}"
                    ) from exc
                if not isinstance(ticket, dict):
                    raise ValueError(
                        f"Invalid checkpoint ticket on line {line_number}"
                    )
                tickets.append(ticket)
        return tickets

    def read_run(self, experiment_id: str, repetition: int) -> dict:
        source = self.run_path(experiment_id, repetition)
        if not source.is_file():
            raise ValueError(f"Finalized repetition {repetition} is missing")
        result = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"Finalized repetition {repetition} is invalid")
        return result

    def write_run(self, experiment_id: str, repetition: int, result: dict) -> Path:
        destination = self.run_path(experiment_id, repetition)
        self._write_json(destination, result)
        return destination

    def append_run_tickets(
        self,
        experiment_id: str,
        repetition: int,
        tickets: list[dict],
    ) -> Path:
        destination = self.partial_run_path(experiment_id, repetition)
        with destination.open("a", encoding="utf-8") as output:
            for ticket in tickets:
                output.write(json.dumps(ticket, ensure_ascii=False))
                output.write("\n")
        return destination

    def finalize_run(
        self,
        experiment_id: str,
        repetition: int,
        result: dict,
    ) -> Path:
        destination = self.run_path(experiment_id, repetition)
        partial = self.partial_run_path(experiment_id, repetition)
        temporary = destination.with_suffix(".json.tmp")
        metadata = dict(result)
        metadata.pop("tickets", None)
        encoded_metadata = json.dumps(metadata, indent=2, ensure_ascii=False)

        with temporary.open("w", encoding="utf-8") as output:
            if metadata:
                output.write(encoded_metadata[:-2])
                output.write(',\n  "tickets": [')
            else:
                output.write('{\n  "tickets": [')
            first = True
            if partial.exists():
                with partial.open(encoding="utf-8") as ticket_input:
                    for line in ticket_input:
                        ticket = line.strip()
                        if not ticket:
                            continue
                        output.write("\n    " if first else ",\n    ")
                        output.write(ticket)
                        first = False
            if not first:
                output.write("\n")
            output.write("  ]\n}\n")

        temporary.replace(destination)
        partial.unlink(missing_ok=True)
        return destination

    def list_runs(self, experiment_id: str) -> list[Path]:
        return sorted(self.directory(experiment_id).glob("run-*.json"))

    def directory(self, experiment_id: str) -> Path:
        try:
            return self._directories[experiment_id]
        except KeyError as exc:
            raise ValueError(f"Unknown experiment {experiment_id}") from exc

    def _update(self, experiment_id: str, **changes) -> None:
        with self._lock:
            status = self._jobs[experiment_id]
            self._jobs[experiment_id] = status.model_copy(update=changes)

    @staticmethod
    def _write_json(destination: Path, payload: dict) -> None:
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
