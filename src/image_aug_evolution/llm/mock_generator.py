from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from image_aug_evolution.augmentation.policy import AugOperation, AugPolicy
from image_aug_evolution.augmentation.search_space import DATASET_CONSTRAINTS, DEFAULT_OPERATION_SPECS


@dataclass
class DatasetHint:
    dataset_name: str
    description: str = ""


class HeuristicPolicyGenerator:
    """Offline stand-in for an LLM policy generator.

    The generator uses dataset hints and prior scores to produce structured
    policies. This keeps the experiment reproducible and runnable without API
    credentials, while preserving the same JSON policy interface that real LLM
    outputs should follow.
    """

    def __init__(self, dataset_name: str, seed: int = 0) -> None:
        self.dataset_name = dataset_name.lower()
        self.rng = random.Random(seed)
        self.ops = list(DEFAULT_OPERATION_SPECS.keys())

    def _weighted_ops(self) -> list[str]:
        discouraged = set(DATASET_CONSTRAINTS.get(self.dataset_name, {}).get("discouraged", []))
        ops = [op for op in self.ops if op not in discouraged]
        if self.dataset_name == "flowers102":
            preferred = ["RandomResizedCrop", "HorizontalFlip", "Rotation", "Affine", "ColorJitter", "RandomErasing", "GaussianBlur"]
        elif self.dataset_name == "eurosat":
            preferred = ["RandomCrop", "HorizontalFlip", "VerticalFlip", "Rotation", "Affine", "GaussianBlur", "ColorJitter"]
        else:
            preferred = ["RandomCrop", "HorizontalFlip", "ColorJitter", "RandomErasing", "Rotation", "Affine", "GaussianBlur"]
        weighted = []
        for op in preferred:
            if op in ops:
                weighted.extend([op] * 3)
        weighted.extend(ops)
        return weighted

    def random_policy(self, policy_id: str, source: str = "heuristic") -> AugPolicy:
        weighted_ops = self._weighted_ops()
        n_sub = self.rng.randint(1, 3)
        sub_policies: list[list[AugOperation]] = []
        for _ in range(n_sub):
            n_ops = self.rng.randint(1, 3)
            chosen = self.rng.sample(weighted_ops, k=min(n_ops, len(weighted_ops)))
            sub_policies.append([
                AugOperation(
                    name=op,
                    probability=round(self.rng.uniform(0.25, 0.95), 2),
                    magnitude=round(self.rng.uniform(0.15, 0.85), 2),
                )
                for op in chosen
            ])
        # Keep mixing sparse; it can be harmful in very small smoke runs.
        mixing = {"mixup_alpha": 0.0, "cutmix_alpha": 0.0}
        if self.rng.random() < 0.25:
            mixing["mixup_alpha"] = round(self.rng.uniform(0.05, 0.4), 2)
        if self.rng.random() < 0.15:
            mixing["cutmix_alpha"] = round(self.rng.uniform(0.05, 0.5), 2)
        return AugPolicy(policy_id=policy_id, sub_policies=sub_policies, mixing=mixing, source=source)

    def initial_population(self, count: int, prefix: str = "heuristic_init") -> list[AugPolicy]:
        return [self.random_policy(f"{prefix}_{i:03d}") for i in range(count)]

    def mutate_from_feedback(self, parent: AugPolicy, policy_id: str, feedback: dict | None = None) -> AugPolicy:
        child = AugPolicy.from_dict(parent.to_dict())
        child.policy_id = policy_id
        child.parent_ids = [parent.policy_id]
        child.source = "heuristic_feedback" if feedback else "pure_evolution_mutation"
        child.notes = f"Mutated from {parent.policy_id}" + (" using validation feedback." if feedback else " without LLM-style feedback.")
        if not child.sub_policies or (len(child.sub_policies) < 4 and self.rng.random() < 0.2):
            child.sub_policies.append([])
        sub = self.rng.choice(child.sub_policies)
        action = self.rng.choice(["adjust", "replace", "add", "remove"])
        weighted_ops = self._weighted_ops()
        if action == "adjust" and sub:
            op = self.rng.choice(sub)
            op.probability = round(min(1.0, max(0.05, op.probability + self.rng.uniform(-0.2, 0.2))), 2)
            op.magnitude = round(min(1.0, max(0.05, op.magnitude + self.rng.uniform(-0.25, 0.25))), 2)
        elif action == "replace" and sub:
            idx = self.rng.randrange(len(sub))
            sub[idx] = AugOperation(
                name=self.rng.choice(weighted_ops),
                probability=round(self.rng.uniform(0.25, 0.9), 2),
                magnitude=round(self.rng.uniform(0.15, 0.85), 2),
            )
        elif action == "add" and len(sub) < 3:
            sub.append(AugOperation(
                name=self.rng.choice(weighted_ops),
                probability=round(self.rng.uniform(0.25, 0.9), 2),
                magnitude=round(self.rng.uniform(0.15, 0.85), 2),
            ))
        elif action == "remove" and len(sub) > 1:
            sub.pop(self.rng.randrange(len(sub)))
        # If performance was unstable, slightly soften mixing.
        if feedback and feedback.get("val_accuracy", 1.0) < feedback.get("population_mean", 0.0):
            child.mixing["mixup_alpha"] = round(max(0.0, float(child.mixing.get("mixup_alpha", 0.0)) * 0.5), 2)
            child.mixing["cutmix_alpha"] = round(max(0.0, float(child.mixing.get("cutmix_alpha", 0.0)) * 0.5), 2)
        return child

    def crossover(self, p1: AugPolicy, p2: AugPolicy, policy_id: str) -> AugPolicy:
        sub_policies = []
        if p1.sub_policies:
            sub_policies.extend(self.rng.sample(p1.sub_policies, k=max(1, len(p1.sub_policies) // 2)))
        if p2.sub_policies:
            sub_policies.extend(self.rng.sample(p2.sub_policies, k=max(1, len(p2.sub_policies) // 2)))
        sub_policies = [[AugOperation(op.name, op.probability, op.magnitude) for op in sub] for sub in sub_policies[:4]]
        mixing = {
            "mixup_alpha": round((float(p1.mixing.get("mixup_alpha", 0.0)) + float(p2.mixing.get("mixup_alpha", 0.0))) / 2, 2),
            "cutmix_alpha": round((float(p1.mixing.get("cutmix_alpha", 0.0)) + float(p2.mixing.get("cutmix_alpha", 0.0))) / 2, 2),
        }
        return AugPolicy(
            policy_id=policy_id,
            sub_policies=sub_policies,
            mixing=mixing,
            source="heuristic_crossover",
            parent_ids=[p1.policy_id, p2.policy_id],
            notes=f"Combined sub-policies from {p1.policy_id} and {p2.policy_id}.",
        )

    @staticmethod
    def _policy_from_ranked_item(item: dict[str, Any] | AugPolicy) -> AugPolicy:
        if isinstance(item, AugPolicy):
            return item
        policy = item.get("policy")
        if isinstance(policy, AugPolicy):
            return policy
        if isinstance(policy, dict):
            return AugPolicy.from_dict(policy)
        raise TypeError(f"Cannot extract policy from ranked item: {item!r}")

    def mutate_from_ranked_population(
        self,
        ranked_population: list[dict[str, Any]],
        policy_id: str,
        feedback: dict | None = None,
    ) -> AugPolicy:
        """Offline analogue of ranked LLM mutation.

        The real OpenAI generator receives the ranked population in a prompt.
        This deterministic fallback selects from the stronger half and mutates
        it, while still exposing the same interface for reproducible tests.
        """
        if not ranked_population:
            return self.random_policy(policy_id, source="heuristic_ranked_fallback")
        stronger_half = ranked_population[len(ranked_population) // 2:] or ranked_population
        parent_item = self.rng.choice(stronger_half)
        parent = self._policy_from_ranked_item(parent_item)
        score_values = [float(item.get("score", 0.0) or 0.0) for item in ranked_population]
        ranked_feedback = {
            **(feedback or {}),
            "population_mean": sum(score_values) / max(1, len(score_values)),
            "parent_score": float(parent_item.get("score", 0.0) or 0.0),
            "ranked_operator": True,
        }
        child = self.mutate_from_feedback(parent, policy_id, ranked_feedback)
        child.source = "heuristic_ranked_mutation"
        child.notes = f"Ranked-population mutation from {parent.policy_id}."
        return child

    def crossover_from_ranked_population(
        self,
        ranked_population: list[dict[str, Any]],
        policy_id: str,
        feedback: dict | None = None,
    ) -> AugPolicy:
        if len(ranked_population) < 2:
            return self.mutate_from_ranked_population(ranked_population, policy_id, feedback)
        top = ranked_population[-min(3, len(ranked_population)):]
        p1_item, p2_item = self.rng.sample(top, k=2)
        p1 = self._policy_from_ranked_item(p1_item)
        p2 = self._policy_from_ranked_item(p2_item)
        child = self.crossover(p1, p2, policy_id)
        child.source = "heuristic_ranked_crossover"
        child.notes = f"Ranked-population crossover from {p1.policy_id} and {p2.policy_id}."
        return child
