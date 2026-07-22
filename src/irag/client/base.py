from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstain": {"type": "boolean"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["answer", "abstain", "evidence_ids", "reason"],
    "additionalProperties": False,
}

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "human_reference_covered": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["human_reference_covered", "confidence", "reason"],
    "additionalProperties": False,
}

GOLD_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "gold_reference_covered": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["gold_reference_covered", "confidence", "reason"],
    "additionalProperties": False,
}


class BaseModelClient(ABC):
    provider: str

    def __init__(self, generation_model: str, auxiliary_model: str) -> None:
        self.generation_model = generation_model
        self.auxiliary_model = auxiliary_model
        self.model_metadata: dict[str, dict[str, Any]] = {}

    @abstractmethod
    def check_models(self) -> None:
        pass

    def decide(self, question: str, retrieved: list[dict]) -> dict[str, Any]:
        system = (
            "Answer only the issue or issues raised in the current SalesX support "
            "ticket. Use only retrieved records that directly support the answer, and "
            "ignore details that address a different issue. From the supporting records "
            "you select, preserve every condition, responsibility, procedure, or "
            "exception that materially changes the answer. Do not combine requirements "
            "from unrelated records. The records are ordered from highest to lowest "
            "relevance after semantic and temporal reranking. If the records are "
            "insufficient or contain a conflict that cannot be resolved from the "
            "available evidence, abstain and return an empty answer. Cite only the "
            "supplied record IDs actually used in evidence_ids."
        )
        user = (
            f"CURRENT TICKET:\n{question}\n\n"
            f"RETRIEVED RECORDS:\n{self._format_context(retrieved)}"
        )
        result, elapsed = self._chat(
            self.generation_model,
            system,
            user,
            ANSWER_SCHEMA,
            schema_name="salesx_decision",
            max_tokens=500,
        )
        result["latency_seconds"] = elapsed
        result["model"] = self.generation_model
        result["provider"] = self.provider
        if result.get("abstain"):
            result["answer"] = ""
            result["evidence_ids"] = []
        return result

    def judge(self, answer: str, human_answer: str) -> dict[str, Any]:
        system = (
            "Assess whether the model answer faithfully covers the human reference for a "
            "SalesX support decision. Return human_reference_covered=true only when every "
            "material outcome, condition, responsibility, procedure, and exception in the "
            "human reference is stated explicitly or clearly entailed by the model answer. "
            "The wording, structure, and level of non-material detail need not be identical, "
            "and relevant elaboration is allowed. Return false if the model answer omits "
            "material reference information, contradicts it, or adds any outcome, "
            "obligation, permission, prohibition, condition, responsibility, procedure, or "
            "exception that materially changes the decision. Treat the human answer only "
            "as the reference; do not use outside knowledge to judge its correctness."
        )
        user = f"MODEL ANSWER:\n{answer}\n\nHUMAN REFERENCE:\n{human_answer}"
        result, elapsed = self._chat(
            self.auxiliary_model,
            system,
            user,
            JUDGMENT_SCHEMA,
            schema_name="salesx_human_reference_coverage",
            max_tokens=300,
        )
        result["latency_seconds"] = elapsed
        result["model"] = self.auxiliary_model
        result["provider"] = self.provider
        return result

    def judge_gold(self, answer: str, gold_answer: str) -> dict[str, Any]:
        system = (
            "Assess whether the candidate answer faithfully covers the gold reference for "
            "a SalesX support decision. Return gold_reference_covered=true only when every "
            "material outcome, condition, responsibility, procedure, and exception in the "
            "gold reference is stated explicitly or clearly entailed by the candidate "
            "answer. The wording, structure, and level of non-material detail need not be "
            "identical, and relevant elaboration is allowed. Return false if the candidate "
            "answer omits material reference information, contradicts it, or adds any "
            "outcome, obligation, permission, prohibition, condition, responsibility, "
            "procedure, or exception that materially changes the decision."
        )
        user = f"CANDIDATE ANSWER:\n{answer}\n\nGOLD REFERENCE:\n{gold_answer}"
        result, elapsed = self._chat(
            self.auxiliary_model,
            system,
            user,
            GOLD_JUDGMENT_SCHEMA,
            schema_name="salesx_gold_reference_coverage",
            max_tokens=250,
        )
        result["latency_seconds"] = elapsed
        result["model"] = self.auxiliary_model
        result["provider"] = self.provider
        return result

    @abstractmethod
    def _chat(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
    ) -> tuple[dict[str, Any], float]:
        pass

    @staticmethod
    def _format_context(retrieved: list[dict]) -> str:
        if not retrieved:
            return "None."
        return "\n\n".join(
            f"RECORD {item['record_id']}\n"
            f"Question: {item['question']}\n"
            f"Final answer: {item['final_answer']}"
            for item in retrieved
        )
