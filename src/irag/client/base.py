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
        "human_equivalent": {"type": "boolean"},
        "gold_equivalent": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["human_equivalent", "gold_equivalent", "confidence", "reason"],
    "additionalProperties": False,
}

GOLD_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "gold_equivalent": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["gold_equivalent", "confidence", "reason"],
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
            "Answer the current SalesX support ticket using only the retrieved prior "
            "question--final-answer records. If those records do not contain sufficient "
            "evidence, abstain and return an empty answer. Otherwise preserve every "
            "material outcome, condition, responsibility, procedure, and exception. "
            "Cite only supplied record IDs in evidence_ids."
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

    def judge(self, answer: str, human_answer: str, gold_answer: str) -> dict[str, Any]:
        system = (
            "Judge semantic equivalence for a SalesX support decision. Equivalent answers "
            "must prescribe the same outcomes, conditions, responsibilities, procedures, "
            "and material exceptions. Compare ANSWER A independently with the human answer "
            "and the gold answer."
        )
        user = (
            f"ANSWER A:\n{answer}\n\n"
            f"HUMAN ANSWER:\n{human_answer}\n\n"
            f"GOLD ANSWER:\n{gold_answer}"
        )
        result, elapsed = self._chat(
            self.auxiliary_model,
            system,
            user,
            JUDGMENT_SCHEMA,
            schema_name="salesx_equivalence_judgment",
            max_tokens=300,
        )
        result["latency_seconds"] = elapsed
        result["model"] = self.auxiliary_model
        result["provider"] = self.provider
        return result

    def judge_gold(self, answer: str, gold_answer: str) -> dict[str, Any]:
        system = (
            "Judge semantic equivalence for a SalesX support decision. Return true only "
            "when both answers prescribe the same outcomes, conditions, responsibilities, "
            "procedures, and material exceptions."
        )
        user = f"ANSWER:\n{answer}\n\nGOLD ANSWER:\n{gold_answer}"
        result, elapsed = self._chat(
            self.auxiliary_model,
            system,
            user,
            GOLD_JUDGMENT_SCHEMA,
            schema_name="salesx_gold_judgment",
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
