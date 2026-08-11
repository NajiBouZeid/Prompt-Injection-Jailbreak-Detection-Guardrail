# Prompt Injection & Jailbreak Detection Guardrail

A real-time guardrail that detects prompt injection and jailbreak attempts before they reach a
target LLM. Three different detection approaches are built and benchmarked against each other on
the same held-out data:

- **Fine-tuned classifier** — a `microsoft/deberta-v3-base` encoder fine-tuned to label a prompt
  `attack` or `benign`.
- **LLM-as-judge** — Gemini (`gemini-flash-lite-latest`) prompted to make the same judgment, no
  training required.
- **LLM Guard baseline** — an existing open-source guardrail library (`llm-guard`'s
  `PromptInjection` scanner), used as an external point of comparison rather than something built
  in this repo.

This project relates to **LLM01: Prompt Injection** in the
[OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

**For the full story, read the two reports in [`docs/`](docs/):**
- **[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)** — a from-scratch, plain-language
  walkthrough of the whole project: the goal, every dataset and model, every decision and why it
  was made, every result and metric, and every bottleneck hit along the way. No prior background
  assumed.
- **[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)** — the full engineering deep-dive:
  architecture, exact code paths, every bug and its fix, all hyperparameters, and the complete
  metrics appendix.

This README is a setup/run guide, not a writeup — see the two reports above for the "why."

## Architecture

```
                              ┌─────────────────────────┐
                              │   3 raw HF datasets      │
                              │   (qualifire, neuralchemy,│
                              │    Necent)                │
                              └────────────┬─────────────┘
                                           │ normalize + dedup
                                           ▼
                              ┌─────────────────────────┐
                              │  data/processed/          │
                              │  combined.parquet         │
                              └────────────┬─────────────┘
                                           │ stratified 80/10/10 split
                                           ▼
                         ┌─────────────────┼─────────────────┐
                         ▼                 ▼                 ▼
                    train.parquet     val.parquet       test.parquet
                         │                 │                 │
                         ▼                 │                 │
         ┌───────────────────────┐         │                 │
         │ Fine-tune DeBERTa-v3   │         │                 │
         │ (src/training)         │◄────────┘ eval_strategy=no │
         └───────────┬───────────┘   (eval runs once, after)  │
                     │ checkpoint-23512                        │
                     ▼                                         │
         ┌───────────────────────┐   ┌─────────────────┐       │
         │ src/evaluation/        │   │ Gemini judge      │      │
         │ evaluate_classifier.py │   │ (gemini_judge.py) │      │
         └───────────┬───────────┘   └─────────┬─────────┘      │
                     │                          │                │
                     │              ┌───────────┴──────────┐     │
                     │              │ LLM Guard baseline     │    │
                     │              │ (isolated venv-llmguard)│   │
                     │              └───────────┬──────────┘     │
                     │                          │                │
                     ▼                          ▼                ▼
              ┌────────────────────────────────────────────────────┐
              │   src/evaluation/compare_judges.py,                 │
              │   compare_three_way.py, threshold_sweep.py,         │
              │   check_leakage.py, check_qualifire_language.py     │
              └───────────────────────────┬──────────────────────────┘
                                          │ picks DEFAULT_THRESHOLD=0.99
                                          ▼
                              ┌─────────────────────────┐
                              │ src/inference/guardrail.py│
                              │       Guardrail class      │
                              └────────────┬─────────────┘
                                           │
                         ┌─────────────────┼─────────────────┐
                         ▼                                   ▼
              demo/backend.py (FastAPI)              scripts/classify.py
              + demo/static/index.html                     (CLI)
                         │
                         ▼
              Dockerfile → containerized demo
```

## Repository layout

```
src/
  config.py              DATASET_NAMES - the 3 HF dataset identifiers used everywhere
  data/
    load_datasets.py     HF download/save/load helpers for raw datasets
    normalize.py         per-dataset schema normalization + Necent scoping + dedup
    split.py             stratified train/val/test split
  training/
    train_classifier.py  DeBERTa fine-tuning (HF Trainer)
  evaluation/
    evaluate_classifier.py   load_model/predict_dataframe/compute_binary_metrics/evaluate()
    gemini_judge.py           LLM-as-judge via Gemini, resumable
    llm_guard_baseline.py     LLM Guard scanner wrapper (run in venv-llmguard)
    compare_judges.py         DeBERTa vs Gemini comparison
    compare_three_way.py      DeBERTa vs Gemini vs LLM Guard comparison
    threshold_sweep.py        decision-threshold sweep
    check_leakage.py          train/test near-duplicate leakage detection
    check_qualifire_language.py  language-contamination check
  inference/
    guardrail.py          Guardrail class - the actual inference path used by the demo/CLI

scripts/                 thin CLI entry points, one per src/ module above, run via `python -m scripts.<name>`
demo/
  backend.py             FastAPI app (loads Guardrail once at startup)
  static/index.html      vanilla HTML/CSS/JS frontend
docs/
  PROJECT_REPORT.md       full plain-language project report
  TECHNICAL_REPORT.md     full technical report
tests/                   (currently empty)
data/                    gitignored - raw/ and processed/ subfolders, regenerable
models/                  gitignored - trained checkpoints, regenerable via training
```

## Setup

### 1. Clone and create a virtual environment

```
git clone https://github.com/NajiBouZeid/Prompt-Injection-Jailbreak-Detection-Guardrail.git
cd Prompt-Injection-Jailbreak-Detection-Guardrail
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

**Important (Windows/NVIDIA GPU users):** the command above installs the CPU-only build of
PyTorch. To use your GPU for training/full-dataset evaluation, reinstall torch with the
CUDA-enabled build:
```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Verify it worked:
```
python -c "import torch; print(torch.cuda.is_available())"
```
This should print `True`.

### 3. (Optional) Set up access for the parts you want to reproduce

These are only needed if you're re-running the corresponding pipeline stage from scratch, not for
using the pre-trained checkpoint/demo:

- **Downloading the raw datasets**: two of the three HF datasets are gated. Run `hf auth login`
  with a HuggingFace account that has been granted access.
- **Gemini judge**: create a free API key at [aistudio.google.com](https://aistudio.google.com),
  then create a `.env` file in the repo root:
  ```
  GEMINI_API_KEY=your-key-here
  ```
  (`.env` is gitignored - never commit it.)
- **LLM Guard baseline**: this library's dependencies conflict with the pinned
  `transformers`/`huggingface-hub` versions the main pipeline needs, so it runs in its own venv:
  ```
  python -m venv venv-llmguard
  venv-llmguard\Scripts\Activate.ps1
  pip install llm-guard pandas pyarrow
  ```

## Running the full pipeline from scratch

Each stage below is a thin CLI wrapper (`scripts/*.py`) over the corresponding logic in `src/`.
Run them in order if reproducing everything from raw data; skip ahead if you already have the
processed data/trained checkpoint.

```
# 1. Download the 3 raw datasets from HuggingFace -> data/raw/*.parquet
python -m scripts.download_datasets

# 2. Normalize + merge + dedup -> data/processed/combined.parquet
python -m scripts.build_processed_dataset

# 3. Stratified 80/10/10 split -> data/processed/{train,val,test}.parquet
python -m scripts.build_splits

# 4. Fine-tune DeBERTa-v3-base (long-running - see docs/TECHNICAL_REPORT.md for
#    real-world throughput and the --resume/--resume-from flags for multi-session runs)
python -m scripts.train_classifier
# to resume an interrupted run:
python -m scripts.train_classifier --resume
# or from a specific checkpoint:
python -m scripts.train_classifier --resume-from models/deberta-v3-base-guardrail/checkpoint-9000

# 5. Evaluate the fine-tuned checkpoint on test/val
python -m scripts.evaluate_classifier --model-dir models/deberta-v3-base-guardrail/checkpoint-23512 --test-split test --output eval_report_test.json
python -m scripts.evaluate_classifier --model-dir models/deberta-v3-base-guardrail/checkpoint-23512 --test-split val --output eval_report_val.json

# 6. Sweep the decision threshold (tuned on val, confirmed on test - see reports for why 0.99 was chosen)
python -m scripts.threshold_sweep --split val --output threshold_sweep_val.json
python -m scripts.threshold_sweep --split test --output threshold_sweep_test.json

# 7. Build the shared 1,439-row sample used by the Gemini/LLM Guard comparisons
python -m scripts.build_gemini_sample

# 8. Run the Gemini judge (resumable across days/quota resets - re-run the same command to continue)
python -m scripts.run_gemini_judge --daily-limit 500

# 9. Run the LLM Guard baseline (must use venv-llmguard's python, not the main venv)
venv-llmguard\Scripts\python.exe -m scripts.run_llm_guard_baseline

# 10. Compare DeBERTa vs Gemini, and all three approaches together
python -m scripts.compare_judges --output eval_report_gemini_comparison.json
python -m scripts.compare_three_way --output eval_report_three_way_comparison.json

# 11. Data-quality checks (train/test leakage, qualifire language contamination)
python -m scripts.check_leakage --model-dir models/deberta-v3-base-guardrail/checkpoint-23512 --output leakage_report_necent_adjusted.json
python -c "
from src.evaluation.check_qualifire_language import run
import json
report, _ = run('models/deberta-v3-base-guardrail/checkpoint-23512')
print(json.dumps(report, indent=2))
"
```

## Using the trained model

### CLI

```
python -m scripts.classify "some prompt to check"
```

### Local web demo

```
python -m scripts.run_demo
```
Then open http://127.0.0.1:8000 in a browser. Everything runs locally; no prompt text is sent
anywhere external.

### Docker (no local Python/venv setup needed)

```
docker build -t prompt-guardrail-demo .
docker run -p 8000:8000 prompt-guardrail-demo
```
Then open http://localhost:8000, same as the local demo above. The image is CPU-only (no GPU
passthrough required), installs from the slimmer `requirements-demo.txt` rather than the full
`requirements.txt`, and only copies the 4 checkpoint files inference actually needs (not the full
training-checkpoint directory). See `docs/TECHNICAL_REPORT.md` for the full rationale.

## Results at a glance

Full numbers, per-dataset breakdowns, and every metric definition are in the two reports. Headline
numbers on the held-out test set (47,024 rows), fine-tuned DeBERTa at the deployed 0.99 threshold:

| Metric | Value |
|---|---|
| Accuracy | 98.49% |
| Precision | 99.82% |
| Recall | 97.78% |
| F1 | 98.79% |
| MCC | 0.968 |

See [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) for the full three-way comparison against
Gemini and LLM Guard, per-dataset breakdowns, calibration analysis, and the data-leakage findings.

## Status

DeBERTa classifier trained and evaluated. Benchmarked against an LLM-as-judge (Gemini) and the
LLM Guard baseline. Local demo and Docker containerization both working end-to-end. See the two
reports in `docs/` for full detail, and their own "what's left" sections for open items.
