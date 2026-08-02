from __future__ import annotations

from dataclasses import dataclass, field

from .policy import AugPolicy
from .search_space import DATASET_CONSTRAINTS, DEFAULT_OPERATION_SPECS


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PolicyValidator:
    def __init__(
        self,
        dataset_name: str = "cifar10",
        max_sub_policies: int = 4,
        max_ops_per_sub_policy: int = 3,
        allow_discouraged: bool = True,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.max_sub_policies = max_sub_policies
        self.max_ops_per_sub_policy = max_ops_per_sub_policy
        self.allow_discouraged = allow_discouraged
        self.operation_specs = DEFAULT_OPERATION_SPECS

    def validate(self, policy: AugPolicy) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if len(policy.sub_policies) > self.max_sub_policies:
            errors.append(f"Too many sub-policies: {len(policy.sub_policies)} > {self.max_sub_policies}")
        for i, sub in enumerate(policy.sub_policies):
            if len(sub) > self.max_ops_per_sub_policy:
                errors.append(f"Sub-policy {i} has too many ops: {len(sub)}")
            for op in sub:
                if op.name not in self.operation_specs:
                    errors.append(f"Unknown operation: {op.name}")
                    continue
                if not (0.0 <= op.probability <= 1.0):
                    errors.append(f"{op.name} probability out of range: {op.probability}")
                spec = self.operation_specs[op.name]
                if not (spec.min_magnitude <= op.magnitude <= spec.max_magnitude):
                    errors.append(f"{op.name} magnitude out of range: {op.magnitude}")
                discouraged = DATASET_CONSTRAINTS.get(self.dataset_name, {}).get("discouraged", [])
                if op.name in discouraged:
                    msg = f"{op.name} is discouraged for {self.dataset_name}"
                    if self.allow_discouraged:
                        warnings.append(msg)
                    else:
                        errors.append(msg)
        for k, value in policy.mixing.items():
            if k not in {"mixup_alpha", "cutmix_alpha"}:
                errors.append(f"Unknown mixing parameter: {k}")
            if not (0.0 <= float(value) <= 2.0):
                errors.append(f"Mixing parameter {k} out of range: {value}")
        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
