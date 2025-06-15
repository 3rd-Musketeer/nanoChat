# Usage

1. Install `uv` from https://docs.astral.sh/uv/

2. Use uv to setup venv and install dependencies

```sh
uv sync
```

3. Configure secret keys

Create `.env` from `.env.example`, fill in the secret keys.

```
# .env

LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL_ID=

DOUBAO_API_KEY=xasdasxxx
DOUBAO_BASE_URL=asdasxxxx
DOUBAO_MODEL_ID=doubao-1.5-vision-lite-250315
```

4. Run demo

```sh
uv run python main.py

# or using venv

source .venv/bin/activate
python main.py
```