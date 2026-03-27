# Japanese Homophone ASR Correction (Standalone)

This package is a minimal standalone subset to run Japanese homophone ASR correction with Qwen3-32B.

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configure API key

Edit:

- `config/test/Japanese/japanese_config_homophones_dialog_call.yaml`

Set:

- `api_key: YOUR_GROQ_API_KEY`

to your real Groq API key.

## 3) Run

```powershell
python .\run_japanese_homophones.py
```

## 4) Main workflow

1. Read ASR input sentence by sentence from `data/japanese/test/test.txt`
2. Tokenize + POS + reading extraction via MeCab
3. Detect suspicious homophone positions from `data/japanese/dictionary/homophones.json`
4. Generate candidates and filter by POS compatibility
5. Score candidates by context/collocation/reading/metadata (and optional embedding signal)
6. Apply rule-based best candidate if confidence/guards pass
7. LLM recheck on uncertain dictionary decisions
8. No-dict fallback LLM for unresolved cases
9. Auto-learn accepted no-dict replacements into runtime/dictionary signals
10. Output corrected text + WER/CER + pipeline stats

## 5) Files included

- `run_japanese_homophones.py` (entrypoint)
- `pipeline_correction.py` (core correction pipeline)
- `tools/compute-wer.py` (WER)
- `tools/compute-cer.py` (CER)
- `tools/japanese_mecab_helper.py` (MeCab wrapper)
- `config/test/Japanese/japanese_config_homophones_dialog_call.yaml` (runtime config)
- `data/japanese/test/test.txt` (input)
- `data/japanese/label/label.txt` (labels)
- `data/japanese/dictionary/homophones.json` (homophone dictionary)

## 6) Notes

- Output is written under `result/`.
- If MeCab dictionary setup fails, install `unidic-lite` and retry.
- This package is intentionally minimal; unrelated scripts and old snapshots are excluded.
