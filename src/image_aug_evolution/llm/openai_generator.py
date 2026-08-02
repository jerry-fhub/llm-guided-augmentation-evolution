from __future__ import annotations

import json
import os
from typing import Any

from image_aug_evolution.augmentation.policy import AugPolicy
from image_aug_evolution.augmentation.search_space import DATASET_CONSTRAINTS, DEFAULT_OPERATION_SPECS
from image_aug_evolution.llm.mock_generator import HeuristicPolicyGenerator


class OpenAIPolicyGenerator:
    """Generate augmentation policies with the OpenAI Responses API.

    This class mirrors the offline heuristic generator interface. It is optional:
    the main experiments remain reproducible without API credentials.
    """

    def __init__(
        self,
        dataset_name: str,
        seed: int = 0,
        model: str = "gpt-5.5",
        temperature: float | None = None,
        max_output_tokens: int = 1800,
        reasoning_effort: str | None = "minimal",
        max_retries: int = 2,
        request_timeout: float | None = None,
        api_mode: str = "responses",
        fallback_to_heuristic: bool = True,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max(1, int(max_retries))
        self.request_timeout = request_timeout
        self.api_mode = api_mode
        self.fallback_to_heuristic = fallback_to_heuristic
        self.heuristic = HeuristicPolicyGenerator(dataset_name, seed=seed)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai Python package is not installed.") from exc
        base_url = os.getenv("OPENAI_BASE_URL")
        kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url or None}
        if self.request_timeout is not None:
            kwargs["timeout"] = float(self.request_timeout)
        self._client = OpenAI(**kwargs)
        return self._client

    def _schema(self, policy_id: str) -> dict[str, Any]:
        operations = sorted(DEFAULT_OPERATION_SPECS.keys())
        operation_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "enum": operations},
                "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "magnitude": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["name", "probability", "magnitude"],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "policy_id": {"type": "string"},
                "source": {"type": "string"},
                "parent_ids": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
                "sub_policies": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": operation_schema,
                    },
                },
                "mixing": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mixup_alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "cutmix_alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["mixup_alpha", "cutmix_alpha"],
                },
            },
            "required": ["policy_id", "source", "parent_ids", "notes", "sub_policies", "mixing"],
        }

    def _system_prompt(self) -> str:
        discouraged = DATASET_CONSTRAINTS.get(self.dataset_name, {}).get("discouraged", [])
        return (
            "You are an expert computer vision researcher and evolutionary search "
            "operator designing data augmentation policies for low-data image "
            "classification. Return only the JSON object that matches the provided "
            "schema. Prefer compact, plausible policies. Use probabilities and "
            "magnitudes conservatively when the dataset is small. When ranked "
            "population examples are provided, infer what separates weak policies "
            "from strong policies, preserve useful baseline priors, and make a "
            "specific improvement rather than random changes. "
            f"Dataset: {self.dataset_name}. Discouraged operations: {discouraged or 'none'}."
        )

    def _call_model(self, policy_id: str, user_prompt: str) -> AugPolicy:
        if self.api_mode == "chat":
            return self._call_chat_model(policy_id, user_prompt)
        if self.api_mode != "responses":
            raise ValueError(f"Unsupported OpenAI API mode: {self.api_mode}")
        client = self._get_client()
        request = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "augmentation_policy",
                    "strict": True,
                    "schema": self._schema(policy_id),
                }
            },
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = None
        text = ""
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                response = client.responses.create(**request)
            except Exception as exc:
                last_error = exc
                request["max_output_tokens"] = int(request["max_output_tokens"]) + 400
                continue
            try:
                text = getattr(response, "output_text", None) or ""
            except Exception as exc:
                last_error = exc
                text = ""
            if not text:
                try:
                    text = self._extract_text(response)
                except Exception as exc:
                    last_error = exc
                    text = ""
            if text:
                break
            request["max_output_tokens"] = int(request["max_output_tokens"]) + 1200
        if not text:
            status = getattr(response, "status", None) if response is not None else None
            incomplete = getattr(response, "incomplete_details", None) if response is not None else None
            raise RuntimeError(
                "OpenAI response did not contain text output "
                f"(status={status}, incomplete_details={incomplete}, last_error={last_error})."
            )
        data = json.loads(text)
        data["policy_id"] = policy_id
        return AugPolicy.from_dict(data)

    @staticmethod
    def _json_from_text(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return json.loads(text)

    def _call_chat_model(self, policy_id: str, user_prompt: str) -> AugPolicy:
        client = self._get_client()
        system_prompt = (
            self._system_prompt()
            + " Return only one raw JSON object. Do not use markdown fences. "
            + "The JSON must contain policy_id, source, parent_ids, notes, sub_policies, and mixing."
        )
        prompt = (
            user_prompt
            + "\n\nRequired JSON shape:\n"
            + json.dumps({
                "policy_id": policy_id,
                "source": "openai_ranked_mutation",
                "parent_ids": [],
                "notes": "short rationale",
                "sub_policies": [[
                    {"name": "RandomCrop", "probability": 1.0, "magnitude": 0.35},
                    {"name": "HorizontalFlip", "probability": 0.5, "magnitude": 0.0}
                ]],
                "mixing": {"mixup_alpha": 0.0, "cutmix_alpha": 0.0},
            }, indent=2)
        )
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": self.max_output_tokens,
                }
                if self.temperature is not None:
                    request["temperature"] = self.temperature
                try:
                    request["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**request)
                except Exception:
                    request.pop("response_format", None)
                    response = client.chat.completions.create(**request)
                text = response.choices[0].message.content or ""
                data = self._json_from_text(text)
                data["policy_id"] = policy_id
                return AugPolicy.from_dict(data)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"OpenAI chat response did not contain valid policy JSON (last_error={last_error}).")

    @staticmethod
    def _extract_text(response: Any) -> str:
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        if not chunks:
            raise RuntimeError("OpenAI response did not contain text output.")
        return "".join(chunks)

    def _safe_call(self, policy_id: str, prompt: str, fallback_source: str) -> AugPolicy:
        try:
            return self._call_model(policy_id, prompt)
        except Exception:
            if not self.fallback_to_heuristic:
                raise
            return self.heuristic.random_policy(policy_id, source=f"{fallback_source}_fallback")

    def random_policy(self, policy_id: str, source: str = "openai_one_shot") -> AugPolicy:
        prompt = (
            f"Create one augmentation policy with policy_id={policy_id!r} for "
            f"{self.dataset_name} low-data image recognition. The source field must be {source!r}."
        )
        policy = self._safe_call(policy_id, prompt, source)
        policy.source = source if policy.source.startswith("openai") else policy.source
        return policy

    def initial_population(self, count: int, prefix: str = "openai_init") -> list[AugPolicy]:
        policies = []
        for i in range(count):
            policy_id = f"{prefix}_{i:03d}"
            prompt = (
                f"Generate initial candidate {i + 1}/{count} for an evolutionary search. "
                f"The policy_id must be {policy_id!r}. Source should be 'openai_initial'."
            )
            policy = self._safe_call(policy_id, prompt, "openai_initial")
            policy.source = "openai_initial" if policy.source.startswith("openai") else policy.source
            policies.append(policy)
        return policies

    def mutate_from_feedback(self, parent: AugPolicy, policy_id: str, feedback: dict | None = None) -> AugPolicy:
        prompt = (
            f"Mutate the parent augmentation policy for {self.dataset_name}. "
            f"The new policy_id must be {policy_id!r}. Source should be 'openai_feedback_mutation'. "
            "Use the validation feedback to improve generalisation without making the policy too strong.\n"
            f"Parent policy JSON:\n{json.dumps(parent.to_dict(), indent=2)}\n"
            f"Feedback JSON:\n{json.dumps(feedback or {}, indent=2)}"
        )
        policy = self._safe_call(policy_id, prompt, "openai_feedback_mutation")
        policy.parent_ids = [parent.policy_id]
        policy.source = "openai_feedback_mutation" if policy.source.startswith("openai") else policy.source
        return policy

    def crossover(self, p1: AugPolicy, p2: AugPolicy, policy_id: str) -> AugPolicy:
        prompt = (
            f"Create a crossover augmentation policy for {self.dataset_name}. "
            f"The new policy_id must be {policy_id!r}. Source should be 'openai_crossover'. "
            "Combine useful elements from both parents while keeping the result concise.\n"
            f"Parent A JSON:\n{json.dumps(p1.to_dict(), indent=2)}\n"
            f"Parent B JSON:\n{json.dumps(p2.to_dict(), indent=2)}"
        )
        policy = self._safe_call(policy_id, prompt, "openai_crossover")
        policy.parent_ids = [p1.policy_id, p2.policy_id]
        policy.source = "openai_crossover" if policy.source.startswith("openai") else policy.source
        return policy

    @staticmethod
    def _ranked_prompt_payload(ranked_population: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = []
        for rank, item in enumerate(ranked_population, start=1):
            policy = item.get("policy")
            if isinstance(policy, AugPolicy):
                policy_json = policy.to_dict()
            else:
                policy_json = policy
            result = item.get("result") or {}
            payload.append({
                "rank_ascending_weak_to_strong": rank,
                "score": item.get("score"),
                "policy": policy_json,
                "metrics": {
                    key: result.get(key)
                    for key in ("val_accuracy", "val_macro_f1", "objective_score", "distortion_proxy", "policy_complexity")
                    if key in result
                },
            })
            for key in (
                "final_pseudo_mask_rate",
                "final_pseudo_confidence",
                "best_pseudo_mask_rate",
                "best_pseudo_confidence",
                "best_unsupervised_loss",
            ):
                if key in result:
                    payload[-1]["metrics"][key] = result.get(key)
        return payload

    def mutate_from_ranked_population(
        self,
        ranked_population: list[dict[str, Any]],
        policy_id: str,
        feedback: dict | None = None,
    ) -> AugPolicy:
        prompt = (
            f"Act as an LLM evolutionary mutation operator for {self.dataset_name}. "
            f"The new policy_id must be {policy_id!r}. Source should be 'openai_ranked_mutation'. "
            "The population below is sorted from weakest to strongest. Create one valid child policy "
            "that is likely to outperform the strongest policies, but keep it conservative enough for "
            "low-data generalisation. Avoid random changes. If baseline_reference is provided, first "
            "identify the strongest baseline method and treat it as the primary prior. If the strongest "
            "baseline is 'none' or 'no_aug', make the child extremely conservative. If it is "
            "'standard_mixup', preserve Mixup and avoid CutMix; if it is 'standard_cutmix', preserve "
            "CutMix and avoid Mixup. Do not add RandomErasing or heavy color/geometric transforms unless "
            "a ranked parent with full-evaluation evidence outperforms the baseline. If the global feedback "
            "says the learning_algorithm is FixMatch, optimize the strong augmentation branch only: set "
            "mixup_alpha and cutmix_alpha to 0, avoid policies that are so weak that pseudo-label consistency "
            "adds little signal, and avoid transformations that are so destructive that they create pseudo-label "
            "noise. Use pseudo mask rate and pseudo confidence as stability evidence when provided.\n"
            f"Ranked population JSON:\n{json.dumps(self._ranked_prompt_payload(ranked_population), indent=2)}\n"
            f"Global feedback JSON:\n{json.dumps(feedback or {}, indent=2)}"
        )
        policy = self._safe_call(policy_id, prompt, "openai_ranked_mutation")
        parent_ids = [
            (item.get("policy").policy_id if isinstance(item.get("policy"), AugPolicy) else item.get("policy", {}).get("policy_id"))
            for item in ranked_population[-3:]
        ]
        policy.parent_ids = [p for p in parent_ids if p]
        policy.source = "openai_ranked_mutation" if policy.source.startswith("openai") else policy.source
        return policy

    def crossover_from_ranked_population(
        self,
        ranked_population: list[dict[str, Any]],
        policy_id: str,
        feedback: dict | None = None,
    ) -> AugPolicy:
        prompt = (
            f"Act as a language-model crossover operator for {self.dataset_name}. "
            f"The new policy_id must be {policy_id!r}. Source should be 'openai_ranked_crossover'. "
            "The population below is sorted from weakest to strongest. Combine complementary useful "
            "parts of the strongest parents, discard harmful transformations seen in weak parents, "
            "and return a compact child policy. If baseline_reference is provided, first identify the "
            "strongest baseline method and treat it as the primary prior. If the strongest baseline is "
            "'none' or 'no_aug', make the child extremely conservative. If it is 'standard_mixup', "
            "preserve Mixup and avoid CutMix; if it is 'standard_cutmix', preserve CutMix and avoid "
            "Mixup. Do not add RandomErasing or heavy color/geometric transforms unless a ranked parent "
            "with full-evaluation evidence outperforms the baseline. If the global feedback says the "
            "learning_algorithm is FixMatch, optimize the strong augmentation branch only: set mixup_alpha "
            "and cutmix_alpha to 0, preserve useful diversity from strong parents, avoid overly weak policies, "
            "and avoid transformations that make pseudo-label consistency noisy. Use pseudo mask rate and "
            "pseudo confidence as stability evidence when provided.\n"
            f"Ranked population JSON:\n{json.dumps(self._ranked_prompt_payload(ranked_population), indent=2)}\n"
            f"Global feedback JSON:\n{json.dumps(feedback or {}, indent=2)}"
        )
        policy = self._safe_call(policy_id, prompt, "openai_ranked_crossover")
        parent_ids = [
            (item.get("policy").policy_id if isinstance(item.get("policy"), AugPolicy) else item.get("policy", {}).get("policy_id"))
            for item in ranked_population[-3:]
        ]
        policy.parent_ids = [p for p in parent_ids if p]
        policy.source = "openai_ranked_crossover" if policy.source.startswith("openai") else policy.source
        return policy
