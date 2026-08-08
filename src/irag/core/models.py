"""Domain models and API request schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Quarter(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    EXTRA = "Extra"

    @property
    def order(self) -> int:
        return {
            Quarter.Q1: 1,
            Quarter.Q2: 2,
            Quarter.Q3: 3,
            Quarter.Q4: 4,
            Quarter.EXTRA: 5,
        }[self]


class Difficulty(str, Enum):
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"


class Category(str, Enum):
    BILLING = "billing"
    INTEGRATIONS = "integrations"
    PERMISSIONS = "permissions"
    REPORTING = "reporting"
    ONBOARDING = "onboarding"


class Profile(str, Enum):
    CEO = "ceo"
    DOMAIN_EXPERT = "domain_expert"
    INTERN = "intern"


class AssignmentStrategy(str, Enum):
    SINGLE = "single_profile"
    RANDOM = "random_mixture"
    INFORMED = "informed_mixture"
    CEO_BOOTSTRAPPED_INFORMED = "ceo_bootstrapped_informed_mixture"


class AcceptanceRegime(str, Enum):
    ALWAYS = "always_accept"
    NEVER = "never_accept"
    STOCHASTIC = "stochastic_50"
    GOLD_SIMILARITY = "gold_similarity"


class SystemState(str, Enum):
    SO = "silent_observer"
    SC = "skeptical_contestator"
    DS = "deferring_surrogate"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    BEDROCK = "bedrock"


class DatasetVariant(str, Enum):
    DRIFT_10 = "drift_10"
    DRIFT_40 = "drift_40"


class ExpertSelection(str, Enum):
    CEO = "ceo"
    DOMAIN_EXPERT = "domain_expert"
    INTERN = "intern"
    RANDOM_MIXTURE = "random_mixture"
    INFORMED_MIXTURE = "informed_mixture"
    CEO_BOOTSTRAPPED_INFORMED_MIXTURE = "ceo_bootstrapped_informed_mixture"


class RunAcceptance(str, Enum):
    ALWAYS_REFUSE = "always_refuse"
    ALWAYS_ACCEPT = "always_accept"
    RANDOMIZE = "randomize"
    GOLD_SIMILARITY = "gold_similarity"


class ProfileAnswer(StrictModel):
    answer: str
    is_correct: bool
    behavior: str | None = None
    applicability: str | None = None
    error_type: str | None = None


class ProfileAnswers(StrictModel):
    ceo: ProfileAnswer
    domain_expert_out_of_domain: ProfileAnswer
    intern: ProfileAnswer


class TicketRecord(StrictModel):
    id: str
    quarter: Quarter
    sequence_in_quarter: int = Field(ge=1, le=500)
    shuffled_order: int = Field(ge=1, le=500)
    difficulty: Difficulty
    category: Category
    policy_key: str
    article_title: str
    question: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    profile_answers: ProfileAnswers
    documentation_anchor: str
    is_changed_answer_near_duplicate: bool
    near_duplicate_of: str | None
    similar_question_ids: list[str]
    drift: dict[str, Any] | None
    generation: dict[str, Any]
    evaluation: dict[str, Any]
    requires_model_abstention: bool = False


class QuarterBatch(StrictModel):
    quarter: Quarter
    records: list[TicketRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_records(self) -> QuarterBatch:
        if any(record.quarter != self.quarter for record in self.records):
            raise ValueError("Every record must match the batch quarter")
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Record IDs must be unique within a quarter")
        return self


class ModelSettings(StrictModel):
    provider: ModelProvider | None = None
    generation_model: str | None = None
    auxiliary_model: str | None = None
    ollama_base_url: str | None = None
    bedrock_region: str | None = None
    timeout: int | None = Field(default=None, ge=1, le=3600)
    retries: int | None = Field(default=None, ge=1, le=10)


class ExperimentCondition(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    assignment_strategy: AssignmentStrategy
    acceptance_regime: AcceptanceRegime = AcceptanceRegime.STOCHASTIC
    single_profile: Profile | None = None
    domain_expert_category: Category = Category.BILLING
    repetitions: int = Field(default=10, ge=1, le=100)
    seed: int = 20260717
    alpha: float = Field(default=0.7, ge=0, le=1)
    beta: float = Field(default=0.55, ge=0, le=1)
    gamma: float = Field(default=0.8, ge=0, le=1)
    minimum_observations: int = Field(default=30, ge=1)
    quarterly_ceo_tickets: int = Field(
        default=100,
        ge=0,
        le=10000,
        validation_alias=AliasChoices(
            "quarterly_ceo_tickets",
            "ds_quarterly_ceo_tickets",
        ),
    )
    top_k: int = Field(default=5, ge=1, le=100)
    semantic_threshold: float = Field(default=0.7, ge=0, le=1)
    lambda_rag: float = Field(
        default=0.99861,
        gt=0,
        le=1,
        description="Insertion-based temporal decay used for RAG reranking.",
    )
    lambda_fea: float = Field(
        default=0.99861,
        gt=0,
        le=1,
        description="Observation-based temporal decay used for FEA updates.",
    )
    recent_gold_window: int = Field(default=30, ge=1, le=1000)
    system_enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_joint_decay(cls, values: Any) -> Any:
        """Read legacy configurations where one lambda controlled both clocks."""
        if not isinstance(values, dict):
            return values
        migrated = dict(values)
        legacy_values = [
            migrated.pop(key)
            for key in ("lambda", "decay")
            if key in migrated
        ]
        if not legacy_values:
            return migrated
        legacy = legacy_values[0]
        if any(value != legacy for value in legacy_values[1:]):
            raise ValueError("Legacy lambda and decay values disagree")
        for field_name in ("lambda_rag", "lambda_fea"):
            if field_name in migrated and migrated[field_name] != legacy:
                raise ValueError(
                    f"{field_name} conflicts with the legacy joint decay value"
                )
            migrated.setdefault(field_name, legacy)
        return migrated

    @model_validator(mode="after")
    def validate_condition(self) -> ExperimentCondition:
        if not self.beta < self.alpha <= self.gamma:
            raise ValueError("Thresholds must satisfy beta < alpha <= gamma")
        if (
            self.assignment_strategy == AssignmentStrategy.SINGLE
            and self.single_profile is None
        ):
            raise ValueError("single_profile is required for single-profile assignment")
        if (
            self.assignment_strategy != AssignmentStrategy.SINGLE
            and self.single_profile is not None
        ):
            raise ValueError(
                "single_profile is only valid for single-profile assignment"
            )
        if (
            not self.system_enabled
            and self.assignment_strategy != AssignmentStrategy.SINGLE
        ):
            raise ValueError("Human-only baselines must use a single profile")
        return self


class ExperimentRequest(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    dataset: DatasetVariant = DatasetVariant.DRIFT_10
    quarters: list[QuarterBatch] = Field(min_length=1, max_length=5)
    conditions: list[ExperimentCondition] = Field(min_length=1)
    models: ModelSettings = Field(default_factory=ModelSettings)

    @model_validator(mode="after")
    def validate_suite(self) -> ExperimentRequest:
        quarter_values = [batch.quarter for batch in self.quarters]
        expected = sorted(quarter_values, key=lambda value: value.order)
        if quarter_values != expected:
            raise ValueError("Quarter batches must be supplied chronologically")
        if len(quarter_values) != len(set(quarter_values)):
            raise ValueError("Quarter batches must be unique")
        record_ids = [record.id for batch in self.quarters for record in batch.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Record IDs must be unique across the experiment")
        condition_names = [condition.name for condition in self.conditions]
        if len(condition_names) != len(set(condition_names)):
            raise ValueError("Condition names must be unique")
        return self


class BundledExperimentRequest(StrictModel):
    name: str = "SalesX bundled experiment"
    dataset: DatasetVariant = DatasetVariant.DRIFT_10
    conditions: list[ExperimentCondition] = Field(min_length=1)
    models: ModelSettings = Field(default_factory=ModelSettings)


class ParallelRunRequest(StrictModel):
    dataset: DatasetVariant = Field(
        default=DatasetVariant.DRIFT_10,
        description="SalesX concept-drift dataset variant.",
    )
    expert: ExpertSelection = Field(
        description="Expert profile or profile-mixture strategy."
    )
    acceptance: RunAcceptance = Field(
        description="How the simulated human handles a conflicting model suggestion."
    )
    repetitions: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Independent repetitions submitted concurrently.",
    )
    lambda_rag: float = Field(
        default=0.99861,
        gt=0,
        le=1,
        description=(
            "Insertion-based RAG decay lambda; the paper decay condition uses "
            "0.99861."
        ),
    )
    lambda_fea: float = Field(
        default=0.99861,
        gt=0,
        le=1,
        description=(
            "Observation-based FEA decay lambda; the paper decay condition uses "
            "0.99861."
        ),
    )
    seed: int = 20260717
    domain_expert_category: Category = Category.BILLING
    include_extra: bool = Field(
        default=False,
        description=(
            "Process the optional 50-ticket post-Q4 abstention challenge. "
            "When false, the run ends after Q4."
        ),
    )
    provider: ModelProvider | None = Field(
        default=None,
        description="Model provider; leave empty to use config.env.",
    )
    generation_model: str | None = Field(
        default=None,
        description="Ticket-decision model; leave empty to use config.env.",
    )
    auxiliary_model: str | None = Field(
        default=None,
        description="Reference-coverage judge; leave empty to use config.env.",
    )
    ollama_base_url: str | None = Field(
        default=None,
        description="Ollama server URL; ignored by other providers.",
    )
    bedrock_region: str | None = Field(
        default=None,
        description="AWS region for Bedrock; leave empty to use config.env.",
    )
    timeout: int | None = Field(
        default=None,
        ge=1,
        le=3600,
        description="Per-request model timeout in seconds.",
    )
    retries: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Model request attempts before failing the run.",
    )
    checkpoint_interval: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Ticket traces written to disk per checkpoint batch.",
    )
    reuse_q1_from: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
        description=(
            "Optional completed run ID whose compatible 500-ticket CEO Q1 "
            "bootstrap should be reused instead of calling the models again."
        ),
    )


class ExperimentCreated(StrictModel):
    experiment_id: str
    status: JobStatus
    status_url: str
    result_url: str


class ExperimentStatus(StrictModel):
    experiment_id: str
    name: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_condition: str | None = None
    completed_repetitions: int = 0
    total_repetitions: int
    output_directory: str
    error: str | None = None
