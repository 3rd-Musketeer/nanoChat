# Usage

## 1. Installation

### 1.1 `uv`

1. Install `uv` from https://docs.astral.sh/uv/

2. Use uv to setup venv and install dependencies

```sh
uv sync
```

### 1.2 `conda`

1. Install `conda` from https://anaconda.org/anaconda/conda

2. Create conda environment

```sh
conda create -n <my_env> python=3.10
pip install .
```

### 1.3 `venv`

```sh
python -m venv .
source .venv/bin/activate
pip install .
```

## 2. Configuration

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

## 3. Usage

### 3.1 Run Realtime Demo

```sh
uv run python main.py

# or using conda
conda activate your_env
python main.py

# or using venv
source .venv/bin/activate
python main.py
```

### 3.2 Run Gradio demo

```sh
uv run python gradio_demo.py

# or using conda
conda activate <your_env>
python gradio_demo.py

# or using venv
source .venv/bin/activate
python gradio_demo.py
```