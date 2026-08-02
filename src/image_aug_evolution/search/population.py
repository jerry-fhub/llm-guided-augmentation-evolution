from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from image_aug_evolution.augmentation.policy import AugPolicy


@dataclass
class PolicyRecord:
    policy: AugPolicy
    generation: int
    validation_ok: bool = False
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    rough_result: dict[str, Any] | None = None
    full_result: dict[str, Any] | None = None
    status: str = "created"

    def score(self, stage: str = "rough") -> float:
        result = self.rough_result if stage == "rough" else self.full_result
        if not result:
            return float("-inf")
        return float(result.get("val_accuracy", float("-inf")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "generation": self.generation,
            "validation_ok": self.validation_ok,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "rough_result": self.rough_result,
            "full_result": self.full_result,
            "status": self.status,
        }
