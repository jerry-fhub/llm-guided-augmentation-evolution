from __future__ import annotations

from image_aug_evolution.llm.mock_generator import HeuristicPolicyGenerator


def build_policy_generator(dataset_name: str, seed: int, config: dict | None = None):
    """Build the configured policy generator.

    The heuristic provider is the default because it is deterministic and does
    not require external credentials. The OpenAI provider is opt-in.
    """
    llm_cfg = (config or {}).get("llm", {})
    provider = str(llm_cfg.get("provider", "heuristic")).lower()
    if provider in {"heuristic", "mock", "offline"}:
        return HeuristicPolicyGenerator(dataset_name, seed=seed)
    if provider == "openai":
        from image_aug_evolution.llm.openai_generator import OpenAIPolicyGenerator

        temperature = llm_cfg.get("temperature")
        return OpenAIPolicyGenerator(
            dataset_name,
            seed=seed,
            model=str(llm_cfg.get("model", "gpt-5.5")),
            temperature=float(temperature) if temperature is not None else None,
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 1800)),
            reasoning_effort=llm_cfg.get("reasoning_effort", "minimal"),
            max_retries=int(llm_cfg.get("max_retries", 2)),
            request_timeout=float(llm_cfg["request_timeout"]) if llm_cfg.get("request_timeout") is not None else None,
            api_mode=str(llm_cfg.get("api_mode", "responses")),
            fallback_to_heuristic=bool(llm_cfg.get("fallback_to_heuristic", True)),
        )
    raise ValueError(f"Unknown LLM policy provider: {provider}")
