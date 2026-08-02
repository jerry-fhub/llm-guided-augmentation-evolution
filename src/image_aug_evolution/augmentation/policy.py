from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AugOperation:
    name: str
    probability: float
    magnitude: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AugOperation":
        return cls(
            name=str(data["name"]),
            probability=float(data.get("probability", 1.0)),
            magnitude=float(data.get("magnitude", 0.5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "probability": self.probability,
            "magnitude": self.magnitude,
        }


@dataclass
class AugPolicy:
    policy_id: str
    sub_policies: list[list[AugOperation]]
    mixing: dict[str, float] = field(default_factory=lambda: {"mixup_alpha": 0.0, "cutmix_alpha": 0.0})
    source: str = "unknown"
    parent_ids: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AugPolicy":
        sub_policies = [
            [AugOperation.from_dict(op) for op in sub_policy]
            for sub_policy in data.get("sub_policies", [])
        ]
        return cls(
            policy_id=str(data.get("policy_id", "policy")),
            sub_policies=sub_policies,
            mixing=dict(data.get("mixing", {"mixup_alpha": 0.0, "cutmix_alpha": 0.0})),
            source=str(data.get("source", "unknown")),
            parent_ids=list(data.get("parent_ids", [])),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "source": self.source,
            "parent_ids": self.parent_ids,
            "notes": self.notes,
            "sub_policies": [[op.to_dict() for op in sub] for sub in self.sub_policies],
            "mixing": self.mixing,
        }

    @property
    def operations(self) -> list[AugOperation]:
        return [op for sub in self.sub_policies for op in sub]


def no_aug_policy(policy_id: str = "no_aug") -> AugPolicy:
    return AugPolicy(policy_id=policy_id, sub_policies=[], source="baseline")
