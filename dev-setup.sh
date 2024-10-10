#!/bin/bash
python3 -m venv .venv #Create a virtual environtment
source .venv/bin/activate
pip install poetry
poetry shell
poetry install
source $(poetry env info --path)/bin/activate # Activate and enter the virtual environment
poetry install --with=dev # Install dev dependencies
pre-commit install --install-hooks --overwrite -t pre-push # Set up pre-commit hooks
export TTI_EVAL_CACHE_PATH=$PWD/.cache
export TTI_EVAL_OUTPUT_PATH=$PWD/output
export HF_TOKEN=<Your_Hugging_Face_API_Token>
