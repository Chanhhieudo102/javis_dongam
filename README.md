# Japanese Homophone ASR Correction

Standalone Japanese homophone correction pipeline using MeCab + Groq LLM (`qwen/qwen3-32b`).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure

Main config file:

- `config/test/Japanese/japanese_config_homophones_dialog_call.yaml`

Required fields:

- `api_key`: your Groq API key
- `model`: `qwen/qwen3-32b`
- `path.test_data` and `path.label`: input/label files

## Run

```powershell
python .\run_japanese_homophones.py
```

## Pipeline Summary

1. Load dictionary and pipeline metadata from `data/japanese/dictionary/homophones.json`.
2. Apply seed replacements from dictionary metadata.
3. Apply verified runtime map replacements.
4. Run dictionary scoring and rule-based candidate replacement.
5. Run LLM recheck for uncertain dictionary decisions.
6. Route unresolved suspicious sentences to no-dict LLM branch.
7. Validate no-dict suggestions with guard checks.
8. Save output text, CER/WER, and detailed `pipeline_stats.json`.

## Outputs

- Run artifacts are generated under `result/pipeline_YYYY-MM-DD_HH-MM-SS/test_data/`.
- Key files include `text`, `wer`, `err`, `runtime.log`, and `pipeline_stats.json`.

## Repository Notes

- `result/` is local runtime output and is excluded from version control.
- `__pycache__/` and `*.pyc` files are excluded from version control.

## Core Files

- `pipeline_correction.py`: correction logic and branch routing
- `run_japanese_homophones.py`: runner and metrics export
- `tools/japanese_mecab_helper.py`: MeCab tokenization helpers
- `tools/compute-wer.py` and `tools/compute-cer.py`: evaluation utilities
