# LLM API Configuration

The project can run in two modes:

- Offline mode uses the deterministic heuristic policy generator and does not need a network connection or API key.
- OpenAI-backed mode uses the OpenAI Python SDK to generate augmentation policies for configs where `llm.provider: openai`.

## Environment Variables

The code reads two environment variables:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

`OPENAI_API_KEY` is required for OpenAI-backed configs. `OPENAI_BASE_URL` is optional. Leave it blank or unset when using the default OpenAI endpoint. Set it only when using an OpenAI-compatible provider or proxy.

Do not commit real credentials. The repository includes `.env.example` only as a safe template, and `.env` is ignored by Git.

## Recommended Local Setup

Create a private `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your own credentials:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

Load it in the current terminal session:

```bash
set -a
source .env
set +a
```

You can also export variables directly without a `.env` file:

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint.example/v1"
```

When using the default OpenAI endpoint, omit `OPENAI_BASE_URL`:

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
unset OPENAI_BASE_URL
```

## Model and API Mode

The model name is configured in the experiment YAML, not through an environment variable:

```yaml
llm:
  provider: openai
  model: gpt-5.5
```

If your provider uses a different model name, edit the `llm.model` field in the config you are running.

The OpenAI generator supports two API modes:

- `api_mode: responses` uses `client.responses.create(...)` and is the default.
- `api_mode: chat` uses `client.chat.completions.create(...)` and is useful for OpenAI-compatible endpoints that do not support the Responses API.

Example:

```yaml
llm:
  provider: openai
  model: your-model-name
  api_mode: chat
  fallback_to_heuristic: false
  max_output_tokens: 900
  max_retries: 2
  request_timeout: 60
```

## Quick Checks

Check that the key is visible to Python without printing the key itself:

```bash
python - <<'PY'
import os
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL") or "default OpenAI endpoint")
PY
```

Run the normal offline smoke suite first:

```bash
python experiments/run_suite.py --suite smoke
```

Then, if credentials are configured, test an OpenAI-backed smoke config:

```bash
python experiments/run_experiment.py \
  --config configs/smoke_openai_ranked_evolution.yaml \
  --mode all
```

## Common Issues

If you see `OPENAI_API_KEY is not set.`, load `.env` in the same terminal where you run the experiment.

If your provider returns endpoint or schema errors, try setting `api_mode: chat` in the config.

If the model name is rejected, change `llm.model` to the exact model identifier supported by your endpoint.
