# nanoChat

Real-time conversational AI with vision capabilities.

## Installation

Choose your preferred environment management tool to set up the project.

### 1. `uv` (Recommended)

1.  Install `uv` from [docs.astral.sh/uv](https://docs.astral.sh/uv).
2.  Create the virtual environment and install dependencies with a single command:
    ```sh
    uv sync
    ```

### 2. `conda`

1.  Install `conda` from the [official website](https://anaconda.org/anaconda/conda).
2.  Create and activate a new conda environment:
    ```sh
    conda create -n nanochat python=3.10
    conda activate nanochat
    ```
3.  Install the required packages using pip:
    ```sh
    pip install -r requirements.txt
    ```

### 3. `pip`

1.  Create and activate a Python virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `.env` file from the example template. If `.env.example` doesn't exist, you can create it manually with the content below.
    ```sh
    cp .env.example .env
    ```
2.  Open the `.env` file and add your secret keys and configuration details.
    ```env
    # .env
    LLM_API_KEY=
    LLM_BASE_URL=
    LLM_MODEL_ID=

    DOUBAO_API_KEY=
    DOUBAO_BASE_URL=
    DOUBAO_MODEL_ID=
    ```

## Usage

### Real-time Demo

Activate your environment and run the main script.

*   Using `uv`:
    ```sh
    uv run python main.py
    ```
*   Using `conda` or `pip`:
    ```sh
    # Make sure your environment (e.g., 'nanochat' or '.venv') is active
    python main.py
    ```

### Gradio Demo

*   Using `uv`:
    ```sh
    uv run python gradio_demo.py
    ```
*   Using `conda` or `pip`:
    ```sh
    # Make sure your environment is active
    python gradio_demo.py
    ```