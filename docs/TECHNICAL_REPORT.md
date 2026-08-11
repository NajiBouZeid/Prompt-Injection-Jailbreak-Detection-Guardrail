# Prompt Injection & Jailbreak Detection Guardrail — Technical Report

*Full engineering detail: architecture, exact code paths, hyperparameters, every bug hit and its
fix, and the complete metrics appendix.*

For the plain-language, no-background-assumed companion to this document, see
[`PROJECT_REPORT.md`](PROJECT_REPORT.md).

---

## Table of contents

1. [System architecture](#1-system-architecture)
2. [Environment & dependency management](#2-environment--dependency-management)
3. [Data pipeline](#3-data-pipeline)
4. [Model training](#4-model-training)
5. [Evaluation infrastructure](#5-evaluation-infrastructure)
6. [LLM-as-judge subsystem (Gemini)](#6-llm-as-judge-subsystem-gemini)
7. [LLM Guard baseline subsystem](#7-llm-guard-baseline-subsystem)
8. [Comparison, threshold-tuning, and data-quality analysis](#8-comparison-threshold-tuning-and-data-quality-analysis)
9. [Inference & serving layer](#9-inference--serving-layer)
10. [Containerization](#10-containerization)
11. [Full bug/gotcha log](#11-full-buggotcha-log)
12. [Complete metrics appendix](#12-complete-metrics-appendix)

---

## 1. System architecture

```
raw HF datasets (3)
  └─ src/data/load_datasets.py .................. download + cache to data/raw/*.parquet
       └─ src/data/normalize.py .................. per-dataset normalization + Necent scoping + dedup
            └─ data/processed/combined.parquet
                 └─ src/data/split.py ............ stratified 80/10/10 split
                      └─ data/processed/{train,val,test}.parquet
                           │
                           ├─► src/training/train_classifier.py .... fine-tune DeBERTa-v3-base
                           │        └─ models/deberta-v3-base-guardrail/checkpoint-23512/
                           │             │
                           │             ├─► src/evaluation/evaluate_classifier.py (full test/val eval)
                           │             ├─► src/evaluation/threshold_sweep.py (decision threshold)
                           │             ├─► src/evaluation/check_leakage.py (train/test leakage)
                           │             └─► src/evaluation/check_qualifire_language.py
                           │
                           ├─► src/evaluation/build_gemini_sample (in scripts/) → 1,439-row shared sample
                           │        ├─► src/evaluation/gemini_judge.py (Gemini judge)
                           │        └─► src/evaluation/llm_guard_baseline.py (venv-llmguard, isolated)
                           │
                           └─► src/evaluation/compare_judges.py, compare_three_way.py
                                    (combine all 3 predictions + Cohen's Kappa)
                                         │
                                         ▼
                                DEFAULT_THRESHOLD = 0.99 adopted
                                         │
                                         ▼
                           src/inference/guardrail.py::Guardrail
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
          demo/backend.py (FastAPI)  scripts/classify.py    Dockerfile
          + demo/static/index.html   (CLI)                  (CPU-only container)
```

## 2. Environment & dependency management

**Two separate virtual environments, deliberately:**
- `venv/` — the main project environment. Python 3.11.9. `torch`, `transformers`, `datasets`,
  `scikit-learn`, `sentence-transformers`, `google-genai`, `fastapi`/`uvicorn`, etc.
- `venv-llmguard/` — isolated environment used *only* to run `llm-guard`. Created after `pip
  install llm-guard` into the main venv silently downgraded pinned core packages (`transformers`
  5.14.1→4.51.3, `huggingface-hub` 1.25.0→0.36.2, `tokenizers` 0.22.2→0.21.4, `cryptography`
  50.0.0→44.0.3) — a downgrade that is dangerous specifically because the fp16-NaN fix (§4) and
  general model-loading behavior are version-sensitive. See [§11](#11-full-buggotcha-log) for the
  full incident.

**GPU setup.** Local NVIDIA RTX 4060 Laptop GPU, 8GB VRAM, driver supports up to CUDA 13.3. Plain
`pip install -r requirements.txt` installs CPU-only torch; GPU support requires reinstalling with
`pip install torch --index-url https://download.pytorch.org/whl/cu124`.

**HuggingFace auth.** Two of the three source datasets are gated on HuggingFace; access requires
`hf auth login` with an approved account. Necent's access request required manual approval
(non-instant); qualifire's was instant despite also being gated.

**Docker demo dependencies** are deliberately *not* the same `requirements.txt` — see
[§10](#10-containerization) for `requirements-demo.txt`'s narrower scope.

## 3. Data pipeline

### `src/config.py`
```python
DATASET_NAMES = [
    "qualifire/prompt-injections-benchmark",
    "neuralchemy/Prompt-injection-dataset",
    "Necent/llm-jailbreak-prompt-injection-dataset",
]
```
Single source of truth for dataset identifiers, imported everywhere a dataset needs to be
referenced by name (download, build, explore scripts).

### `src/data/load_datasets.py`
`load_hf_dataset(name)` / `save_raw_dataset(name, df)` / `load_raw_dataset(name)` — thin wrappers
around HuggingFace's `datasets` library and local parquet caching under `data/raw/`, so the
pipeline never re-downloads from HF once a dataset has been fetched once.

### `src/data/normalize.py`
Per-dataset normalizer functions, each producing the same 4-column schema
(`text`, `label`, `source_dataset`, `category`):

```python
def normalize_qualifire(df):
    return pd.DataFrame({
        "text": df["text"],
        "label": (df["label"] == "jailbreak").astype(int),
        "source_dataset": "qualifire",
        "category": df["label"],
    })

def normalize_neuralchemy(df):
    return pd.DataFrame({
        "text": df["text"],
        "label": df["label"].astype(int),
        "source_dataset": "neuralchemy",
        "category": df["category"],
    })

NECENT_SCOPED_PROMPT_TYPES = ["prompt_injection", "jailbreak"]

def normalize_necent(df):
    scoped = df[df["prompt_type"].isin(NECENT_SCOPED_PROMPT_TYPES)]
    return pd.DataFrame({
        "text": scoped["prompt"],
        "label": scoped["prompt_adversarial"].astype(int),
        "source_dataset": "necent",
        "category": scoped["prompt_type"],
    })
```

Key decision baked into `normalize_necent`: Necent aggregates 30+ datasets covering many unrelated
safety problems (toxicity, harmful-behavior requests, generic content moderation). Only rows
tagged `prompt_injection` or `jailbreak` are kept — everything else is dropped at this stage,
before any further processing. `prompt_adversarial` (identical to `is_dangerous` within this
scope) becomes the unified `label`.

`build_combined_dataset(raw_dfs)` concatenates all three normalized frames, strips whitespace,
drops empty-text rows, and deduplicates on exact `text` match (`keep="first"`, so ordering in
`DATASET_NAMES` determines which source "wins" a collision — qualifire/neuralchemy before necent).

### `src/data/split.py`
```python
def stratified_split(df, val_size=0.1, test_size=0.1, random_state=42):
    strata = df["source_dataset"] + "_" + df["label"].astype(str)
    train_df, temp_df = train_test_split(df, test_size=val_size+test_size,
                                          stratify=strata, random_state=random_state)
    temp_strata = strata.loc[temp_df.index]
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(val_size+test_size),
                                        stratify=temp_strata, random_state=random_state)
    return train_df, val_df, test_df
```
Two-stage `sklearn.model_selection.train_test_split`, stratified on the *combined* `source_dataset
+ label` key (not label alone) — this is what guarantees every split gets a representative slice of
qualifire/neuralchemy despite their tiny size relative to necent. `random_state=42` throughout for
reproducibility.

### Resulting dataset statistics (measured directly from `data/processed/*.parquet`)

| Split | Rows | necent | qualifire | neuralchemy | Label balance (1/0) | Text length (mean/median chars) |
|---|---|---|---|---|---|---|
| train | 376,185 | 368,672 | 4,000 | 3,513 | 63.06% / 36.94% | 544.9 / 375.0 |
| val | 47,023 | 46,084 | 500 | 439 | 63.06% / 36.94% | 555.0 / 374.0 |
| test | 47,024 | 46,085 | 500 | 439 | 63.06% / 36.94% | 544.3 / 375.0 |
| combined | 470,232 | 460,841 | 5,000 | 4,391 | 63.06% / 36.94% | 545.8 / 375.0 |

Max text length: 55,089 characters (one outlier row in train); min: 10 characters.

## 4. Model training

### `src/training/train_classifier.py`

```python
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, dtype=torch.float32
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,      # effective batch size 16
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="no",                 # see fix rationale below
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    bf16=True,
    logging_steps=100,
    report_to="none",
)
```

Tokenization: head-truncation to `MAX_LENGTH=512` tokens (HF's default single-sequence behavior:
keep the first 512 tokens, drop the rest), dynamic per-batch padding via
`DataCollatorWithPadding` (not fixed-length padding, since most texts are far shorter than 512
tokens).

**fp16 NaN bug.** `deberta-v3-base`'s HF checkpoint is stored in fp16, and the transformers version
in use (5.14.1) defaults to loading models in their checkpoint's native dtype. DeBERTa-v3's
disentangled attention (`XSoftmax`) produces NaN logits under pure fp16. **Fix**: load master
weights explicitly as `dtype=torch.float32`, and use `bf16=True` in `TrainingArguments` for mixed
precision during the actual forward/backward pass instead of `fp16=True` (the RTX 4060 supports
bf16 natively).

**`eval_strategy="no"` rationale.** Originally `eval_strategy="steps"` matched `save_strategy`
(both every 1000 steps), enabling `load_best_model_at_end`. But a full pass over the 47,023-row val
set took **~55 minutes per checkpoint** on this hardware — with ~23 checkpoints expected in one
epoch, that's ~21 *additional* hours purely from per-checkpoint eval. Fixed by disabling
per-step eval entirely and calling `trainer.evaluate()` once, explicitly, after `trainer.train()`
finishes. `load_best_model_at_end`/`metric_for_best_model` were removed accordingly (no per-step
eval to compare checkpoints against anymore).

**Checkpointing for multi-session training.** `save_steps=1000`, `save_total_limit=3` (keeps 3
most-recent checkpoints on disk). `scripts/train_classifier.py` exposes:
```
python -m scripts.train_classifier --resume                              # auto-detect latest checkpoint
python -m scripts.train_classifier --resume-from CHECKPOINT_PATH          # explicit path
```
`--resume-from` was added specifically after the checkpoint-2000 corruption incident (§11) — a
deliberate way to bypass a corrupted "latest" checkpoint and resume from a known-good earlier one.

**Final training result.** Full epoch completed: step 23,512/23,512, `train_runtime` ≈ 25,170s
(~7h), `train_samples_per_second` 14.95, **final `train_loss` 0.04298**. Final model saved to
`models/deberta-v3-base-guardrail/checkpoint-23512/` — verified complete (all 9 expected files
present, `optimizer.pt` full size ~1.4GB). Disk footprint of this final checkpoint directory:

| File | Size |
|---|---|
| `model.safetensors` | 737,719,272 bytes (~704MB) |
| `optimizer.pt` | 1,475,558,394 bytes (~1.4GB, training-only state) |
| `tokenizer.json` | 8,339,974 bytes |
| `config.json`, `tokenizer_config.json`, `trainer_state.json`, `training_args.bin`, `rng_state.pth`, `scheduler.pt` | small (<50KB each) |

Only `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` are needed for
*inference* — the rest (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`,
`training_args.bin`) is training-only resumption state, ~1.42GB of dead weight for a deployed
model. This distinction is exploited directly in the Docker build (§10).

## 5. Evaluation infrastructure

### `src/evaluation/evaluate_classifier.py`

Core reusable functions, used by nearly every other evaluation/comparison script in the project:

```python
def load_model(model_dir: str) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32
    ).to(device)
    model.eval()
    return model, tokenizer, device

def predict_dataframe(model, tokenizer, df, device, batch_size=32) -> tuple[np.ndarray, np.ndarray]:
    # tokenize + DataLoader + batched forward pass, returns (hard predictions, positive-class probs)
```

`predict_dataframe` explicitly excludes the `label` column from the tokenized/collated dataset
(`remove_columns=["text", "label"]`, `set_format(columns=["input_ids","attention_mask"])`) — see
the KeyError bug in §11 for why this matters.

`compute_binary_metrics(labels, predictions, probs=None)` — the single metrics function used
project-wide, computing:
```python
def compute_binary_metrics(labels, predictions, probs=None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    accuracy = (predictions == labels).mean()
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": ..., "precision": ..., "recall": ..., "f1": ...,
        "mcc": matthews_corrcoef(labels, predictions),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": len(labels),
    }
    if probs is not None and len(np.unique(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
        metrics["pr_auc"] = average_precision_score(labels, probs)
        metrics["brier_score"] = brier_score_loss(labels, probs)   # added this session
        metrics["ece"] = expected_calibration_error(labels, probs)  # added this session
    return metrics
```

**Calibration metrics added this session** (Brier score, ECE) — not present in earlier project
sessions, added specifically to quantify the qualifire calibration finding numerically rather than
inferring it only from the ROC-AUC/accuracy gap. `expected_calibration_error()`:

```python
def expected_calibration_error(labels, probs, n_bins=10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_confidence - bin_accuracy)
    return float(ece)
```
Standard 10-bin, equal-width ECE formulation: mean |confidence - accuracy| gap, weighted by bin
population.

`measure_latency(model, tokenizer, texts, device, n_samples=200)` — single-example (batch size 1)
forward-pass timing, deliberately separate from the batched `predict_dataframe` throughput, with
`torch.cuda.synchronize()` called before stopping the timer on CUDA devices (otherwise the GPU call
returns asynchronously and the measured time would be meaninglessly short).

`evaluate(model_dir, test_split, batch_size, max_examples, latency_samples)` — the top-level
orchestrator: loads the model, runs `predict_dataframe` over the full split, computes overall +
per-`source_dataset` metrics (via `df_eval.groupby("source_dataset")`), computes eval loss
(cross-entropy from probabilities, comparable to HF's training loss), and runs the latency
benchmark. Returns one JSON-serializable dict.

**CLI**: `python -m scripts.evaluate_classifier --model-dir <path> [--test-split test|val] [--max-examples N] [--output report.json]`

## 6. LLM-as-judge subsystem (Gemini)

### `src/evaluation/gemini_judge.py`

```python
JUDGE_INSTRUCTION = (
    "You are a security classifier for an LLM guardrail. You will be shown a "
    "piece of text that a user submitted to an AI assistant. Decide whether it "
    "is a prompt injection or jailbreak attack ... Respond with only the classification."
)
RESPONSE_SCHEMA = {"type": "OBJECT", "properties": {"label": {"type": "STRING", "enum": ["attack", "benign"]}}, "required": ["label"]}
SAFETY_SETTINGS = [types.SafetySetting(category=cat, threshold="BLOCK_ONLY_HIGH") for cat in [...]]
```

Structured JSON output (`response_schema` forces `{"label": "attack"|"benign"}`), `temperature=0`
for determinism, safety thresholds loosened to `BLOCK_ONLY_HIGH` on all 4 categories — otherwise
Gemini's default safety filter can trip on the *classification input itself* (attack/jailbreak
text), which defeats the purpose of asking it to classify that exact text.

`judge_one(client, model, text)` — one API call, returns `{"label": ...}` or, if
`response.candidates`/`response.text` is empty, `{"label": None, "block_reason": ...}` extracted
from `response.prompt_feedback.block_reason`.

**Resumability design** (explicit requirement — "checkpoints" mid-run):
```python
def load_done_ids(output_path) -> set[int]:
    # reads existing row_ids from output_path, so already-judged rows are skipped on any rerun
```
Every row's result is written to `data/processed/gemini_judge_results.jsonl` and `f.flush()`ed
**immediately** after each API call, not batched — there is no meaningful concept of "checkpoint at
row 500," it's safe to stop the process at literally any point without losing prior work.

`run_judge(model="gemini-flash-lite-latest", output_path=..., rpm=15, daily_limit=1000)` — paces
requests at `60/rpm * 1.1` seconds between calls (small safety margin under the RPM cap), catches
`RESOURCE_EXHAUSTED` exceptions specifically and `break`s the loop cleanly (see quota bug in §11
for why this exists as a specific exception check rather than a generic catch-all).

### `scripts/build_gemini_sample.py`
```python
NECENT_SAMPLE_SIZE = 500
RANDOM_SEED = 42
qualifire = df[df["source_dataset"] == "qualifire"]      # all 500 rows
neuralchemy = df[df["source_dataset"] == "neuralchemy"]  # all 439 rows
necent = df[df["source_dataset"] == "necent"].sample(n=500, random_state=42)
sample = pd.concat([qualifire, neuralchemy, necent]).sample(frac=1, random_state=42)
```
Deliberately non-proportional sampling: full coverage of the two smaller/curated datasets (where
DeBERTa's known weak spot lives), a 500-row check on necent (huge and already near-perfect, so a
smaller sample suffices). Output: `data/processed/gemini_judge_sample.parquet`, 1,439 rows,
`row_id` = the parquet's row index, deterministic via `random_state=42`.

**CLI**: `python -m scripts.run_gemini_judge --model gemini-flash-lite-latest --daily-limit 500`
(default model was buggy earlier — see §11).

## 7. LLM Guard baseline subsystem

### `src/evaluation/llm_guard_baseline.py`

```python
from llm_guard.input_scanners import PromptInjection
scanner = PromptInjection()   # default model: protectai/deberta-v3-base-prompt-injection-v2
_, is_valid, risk_score = scanner.scan(text)
predicted_label = 0 if is_valid else 1
```

`PromptInjection.__init__` signature: `(self, *, model=None, threshold=0.92, match_type=MatchType.FULL, use_onnx=False)`
— default threshold 0.92, not overridden in this project (baseline run with library defaults, to
represent what an out-of-the-box integration would actually see). `scan()` returns
`(sanitized_prompt, is_valid, risk_score)`. Quirk observed: `risk_score` is `-1.0` (not `0.0`) on a
clean/valid prompt — stored as-is, not treated as a bug, just a scale idiosyncrasy of this
particular library (meaning `risk_score` is *not* a calibrated 0-1 probability and isn't used in
any Brier/ECE calculation — those metrics are only computed for DeBERTa's true softmax
probabilities in this project).

Must run under **`venv-llmguard`**, never the main venv (dependency conflict, §11). Runs on CPU in
this isolated environment (no GPU torch installed there — acceptable for a one-off baseline run).

**Resumable**, same JSONL-append pattern as `gemini_judge.py`: `load_done_ids()` /
per-row-immediate-flush. LLM Guard has no refusal concept (unlike Gemini) — every row of the
1,439-row sample always gets scored.

**CLI**: `venv-llmguard\Scripts\python.exe -m scripts.run_llm_guard_baseline`

## 8. Comparison, threshold-tuning, and data-quality analysis

### `src/evaluation/compare_judges.py` (DeBERTa vs Gemini)
Loads the shared sample, runs DeBERTa via `predict_dataframe`, loads Gemini's JSONL results,
excludes Gemini's refused rows (`gemini_refused = predicted_label.isna()`) from Gemini's own
metrics but reports `n_refused`/`n_total` per source separately. Computes `agreement_rate` (raw %
match on rows Gemini answered) and, added this session, `cohen_kappa` via
`sklearn.metrics.cohen_kappa_score` on the same answered subset. Also emits a full
`disagreements` list (row_id, source, true_label, both predictions, 200-char text preview) for
manual qualitative review.

### `src/evaluation/compare_three_way.py` (DeBERTa vs Gemini vs LLM Guard)
Extends the same pattern to three judges. Added this session: `pairwise_agreement` block with
Cohen's Kappa + raw agreement rate for all three judge pairs, each computed on the appropriate
subset (any pair including Gemini excludes Gemini's refused rows; DeBERTa vs LLM Guard uses all
1,439 rows since neither refuses):
```python
report["pairwise_agreement"] = {
    "deberta_vs_gemini": {"cohen_kappa": ..., "raw_agreement_rate": ..., "n": 1433},
    "deberta_vs_llm_guard": {"cohen_kappa": ..., "raw_agreement_rate": ..., "n": 1439},
    "gemini_vs_llm_guard": {"cohen_kappa": ..., "raw_agreement_rate": ..., "n": 1433},
}
```

### `src/evaluation/threshold_sweep.py`
`sweep(model_dir, split, batch_size)` — runs `predict_dataframe` once, then sweeps a fixed decision
threshold across the already-computed probabilities (no re-inference per threshold — cheap).
Swept in three widening passes (0.05-0.95 step 0.05 → 0.90-0.99 step 0.01 → 0.99-0.9999 step
0.002) because the optimum kept landing on the edge of the tested range each time. Tuned on `val`,
confirmed on `test` (genuinely held out from the sweep itself). `best_threshold_for_source(report,
source, metric)` finds the threshold maximizing a given metric for a given source slice.

**CLI**: `python -m scripts.threshold_sweep --split val --output threshold_sweep_val.json`

### `src/evaluation/check_leakage.py`
Two independent near-duplicate detection methods against necent train (368,672 rows) vs. a
2,000-row random sample of necent test:
- **Embedding method**: `sentence-transformers`' `all-MiniLM-L6-v2`, normalized embeddings,
  chunked (20k-row chunks) matrix-multiply cosine similarity — max similarity per test row.
- **MinHash/Jaccard method**: word 5-shingles, `datasketch.MinHashLSH` (`num_perm=128,
  threshold=0.5`) for fast candidate lookup, exact Jaccard computed only among LSH candidates.

`run(train_path, test_path, test_sample_size, model_dir=None)` — when `model_dir` is set, also
scores DeBERTa on the same 2,000-row sample and adds `leakage_adjusted_metrics`: three
`compute_binary_metrics` breakdowns for `leaked_embedding_ge_0.99`, `leaked_embedding_ge_0.95`, and
`clean_embedding_lt_0.95` subsets.

**CLI**: `python -m scripts.check_leakage --model-dir <checkpoint> --output leakage_report_necent_adjusted.json`

### `src/evaluation/check_qualifire_language.py`
```python
DetectorFactory.seed = 42   # langdetect is non-deterministic run-to-run unless seeded
```
Runs DeBERTa on all 500 qualifire test rows, detects language per row via `langdetect`, cross-tabs
detected language against tp/tn/fp/fn outcome. No CLI wrapper exists (run ad hoc via a small
`python -c` snippet, documented in the README) — this was an exploratory one-off check, not part of
the regular pipeline.

## 9. Inference & serving layer

### `src/inference/guardrail.py`
```python
DEFAULT_MODEL_DIR = "models/deberta-v3-base-guardrail/checkpoint-23512"
DEFAULT_THRESHOLD = 0.99   # adopted via the threshold sweep, see §8

class Guardrail:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, threshold=DEFAULT_THRESHOLD):
        self.model, self.tokenizer, self.device = load_model(model_dir)  # reuses evaluate_classifier.load_model
        self.threshold = threshold

    def classify(self, text: str) -> dict:
        inputs = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()[0]
        exp_logits = np.exp(logits - logits.max())
        prob_attack = float((exp_logits / exp_logits.sum())[1])
        return {"label": "attack" if prob_attack >= self.threshold else "benign",
                "prob_attack": prob_attack, "threshold": self.threshold}
```
This is the first code path in the repo that actually *applies* the 0.99 threshold decision, as
opposed to just measuring it in an eval report. Loaded once (not per-request) by anything that
constructs a `Guardrail` instance.

### `demo/backend.py`
```python
app = FastAPI()
guardrail = Guardrail()                                   # loaded once at process startup
test_df = pd.read_parquet("data/processed/test.parquet")  # loaded once at process startup

@app.post("/api/classify")
def classify(request: ClassifyRequest):
    start = time.perf_counter()
    result = guardrail.classify(request.text)
    result["latency_ms"] = (time.perf_counter() - start) * 1000
    return result

@app.get("/api/sample")
def sample():
    row = test_df.sample(1).iloc[0]
    return {"text": row["text"], "true_label": "attack" if row["label"] == 1 else "benign",
            "source_dataset": row["source_dataset"]}

app.mount("/", StaticFiles(directory="demo/static", html=True), name="static")  # mounted LAST
```
**Route-ordering gotcha (intentional, documented in-code)**: `StaticFiles` is mounted at `/` *after*
`/api/classify`/`/api/sample` are registered — FastAPI matches routes in registration order, so
mounting the catch-all static handler first would shadow the API routes entirely.

### `demo/static/index.html`
Self-contained vanilla HTML/CSS/JS single-page frontend, no build step, no external dependencies.
Deliberately reuses the exact color tokens and card/callout visual language from the published
eval-report artifact (`--accent: #2a78d6` light / `#3987e5` dark, same surface/border/text token
structure) for visual consistency between the report and the live demo. Three-state theming: a
`data-theme` attribute (manual toggle) overrides `prefers-color-scheme` (system default), which
overrides the light-mode `:root` defaults. Features: textarea input, 4 cycling example
attack/benign chips, a "🎲 Sample from test set" chip hitting `/api/sample`, a probability bar with
the 0.99 threshold rendered as a visual marker line on the track, latency display.

### `scripts/run_demo.py`
```python
def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("demo.backend:app", host=host, port=port)
```
Defaults preserve normal local (non-Docker) behavior; the Docker image (§10) overrides via
`ENV HOST=0.0.0.0` so the server is reachable from outside the container through `-p` port mapping
— binding to `127.0.0.1` inside a container is unreachable from the host even with port mapping,
which is why this needed to become configurable rather than hardcoded.

### `scripts/classify.py`
Thin CLI: `python -m scripts.classify "some prompt to check" [--model-dir ...] [--threshold ...]`,
constructs a `Guardrail` and prints the label/probability/threshold.

## 10. Containerization

### `requirements-demo.txt`
A deliberately narrower dependency set than the main `requirements.txt`:
```
transformers
datasets
scikit-learn
pandas
numpy
pyarrow
sentencepiece
protobuf
fastapi
uvicorn
```
Excludes training/eval-only packages the demo never imports at runtime: `sentence-transformers`,
`google-genai`, `python-dotenv`, `datasketch`, `langdetect`, `accelerate`. `torch` is deliberately
*absent* from this file — installed separately in the Dockerfile via
`--index-url https://download.pytorch.org/whl/cpu`, since the default PyPI `torch` wheel bundles a
multi-GB CUDA build that provides no benefit in a CPU-only container. `datasets`/`scikit-learn` are
kept even though `demo/backend.py` never calls them directly, because `Guardrail` →
`evaluate_classifier.py` imports them at module level (`from datasets import Dataset`, `from
sklearn.metrics import ...`) — importing the module pulls in its full import graph regardless of
which functions are actually called.

### `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY src/ src/
COPY scripts/run_demo.py scripts/run_demo.py
COPY demo/ demo/

COPY models/deberta-v3-base-guardrail/checkpoint-23512/config.json \
     models/deberta-v3-base-guardrail/checkpoint-23512/model.safetensors \
     models/deberta-v3-base-guardrail/checkpoint-23512/tokenizer.json \
     models/deberta-v3-base-guardrail/checkpoint-23512/tokenizer_config.json \
     models/deberta-v3-base-guardrail/checkpoint-23512/

COPY data/processed/test.parquet data/processed/test.parquet

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000
CMD ["python", "-m", "scripts.run_demo"]
```
Only the 4 inference-relevant checkpoint files are copied (multi-source `COPY` requires the
destination to end in `/`) — `optimizer.pt`/`scheduler.pt`/`rng_state.pth`/`trainer_state.json`/
`training_args.bin` (~1.42GB of training-only state, see §4) are never copied into the image.

### `.dockerignore`
```
models/**
!models/deberta-v3-base-guardrail/
!models/deberta-v3-base-guardrail/checkpoint-23512/
!models/deberta-v3-base-guardrail/checkpoint-23512/config.json
!models/deberta-v3-base-guardrail/checkpoint-23512/model.safetensors
!models/deberta-v3-base-guardrail/checkpoint-23512/tokenizer.json
!models/deberta-v3-base-guardrail/checkpoint-23512/tokenizer_config.json
```
Explicit re-inclusion pattern required here: a bare `models/` blanket-ignore would also hide these
4 specific files from the Docker build context entirely (ignored paths never reach the daemon), so
each parent directory level must be explicitly un-ignored before the specific files can be
re-included.

**Final image**: `prompt-guardrail-demo:latest`, **3.74GB**. Verified end-to-end by running the
actual container (`docker run -p 8001:8000 ...`) and `curl`-testing all three surfaces: an attack
prompt (`POST /api/classify` → `{"label":"attack","prob_attack":0.99998,...}`), a benign prompt
(→ `{"label":"benign","prob_attack":0.00226,...}`), `GET /api/sample` (returned a real labeled test
row), and `GET /` (200, static frontend) — not just a successful `docker build`.

**Network incident during this build**: `docker build` initially failed twice with `DeadlineExceeded:
failed to fetch oauth token` reaching `auth.docker.io`. Diagnosis, in order: (1) confirmed Docker
Desktop's daemon itself wasn't running (`docker info` failed with a named-pipe error) — started it
via `Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"`, waited for `docker info`
to succeed; (2) a second, distinct failure turned out to be a genuine network issue — host-level
`curl` also timed out reaching `auth.docker.io` *and* `github.com` specifically, while
`google.com`, `huggingface.co`, and `pypi.org` all responded normally. DNS server resolved to
`172.20.10.1` (the standard iPhone Personal Hotspot gateway address), consistent with a
mobile-hotspot connection selectively failing to route to those two specific hosts. Confirmed not
transient with several retries before concluding it was network-side rather than a Docker
misconfiguration; resolved once the connection stabilized and the build succeeded on retry.

## 11. Full bug/gotcha log

A consolidated technical log of every bug hit across the project, for anyone extending this
codebase later:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `.gitignore` accidentally matched `src/data/` | Unanchored `data/`/`models/` patterns matched any directory named `data`/`models` anywhere in the tree, not just at repo root | Anchored patterns to `/data/` and `/models/` (root-only) |
| 2 | Training `eval_loss` showed `nan` | `deberta-v3-base`'s checkpoint stored in fp16; its disentangled attention (`XSoftmax`) produces NaN logits under pure fp16 | Load weights as `dtype=torch.float32` explicitly; use `bf16=True` in `TrainingArguments` for mixed precision instead of `fp16=True` |
| 3 | Script hangs silently at 0% CPU right after "Loading weights" | A killed process mid-download left a stale `.lock`/`.incomplete` file in `~/.cache/huggingface/hub/.locks/...`, blocking all future loads of that model | Check for and remove stale lock/incomplete files in the HF cache |
| 4 | Early training-time estimates (4-5h, then 14h per epoch) were both wrong | Short smoke tests and even a couple hours of real training don't reflect true steady-state throughput | Estimate only from a large, multi-hour real sample; the actual ~23-24h/epoch (before the eval-strategy fix) only became visible after 2 real hours |
| 5 | Per-checkpoint eval added ~21 hours on top of training | `eval_strategy="steps"` matched `save_strategy` (every 1000 steps); a full 47k-row val pass took ~55 min per checkpoint × ~23 checkpoints | `eval_strategy="no"`; call `trainer.evaluate()` once after `trainer.train()` finishes |
| 6 | checkpoint-2000 failed to resume with `FileNotFoundError` on `trainer_state.json` | The prior session's process was stopped right after `nvidia-smi` showed 0% GPU util, but the checkpoint save was still mid-write to disk — GPU-idle does not guarantee a checkpoint finished writing | Verify a checkpoint has all 9 expected files (esp. `trainer_state.json`, `optimizer.pt` at full ~1.4GB size) before trusting it; added `--resume-from CHECKPOINT_PATH` to bypass a corrupted "latest" checkpoint |
| 7 | `SafetensorError: not enough space on the disk` mid-checkpoint-save | HF's `Trainer` saves the new checkpoint *before* deleting the oldest under `save_total_limit`, causing a transient peak of ~4 checkpoints (~8.4GB) on disk right after each save; unrelated other processes on the same machine had also filled the disk to 96-100% | Freed space from unrelated HF cache entries; understood the transient-peak behavior as expected, not a bug |
| 8 | `evaluate_classifier.py` crashed with `KeyError: 'label'` inside the batch loop | `DataCollatorWithPadding` (via `tokenizer.pad()`) silently drops any column not among its known tokenizer keys — a column literally named `label` (not HF's expected `labels`) never reaches the batch dict | Excluded `label` from tokenized/collated columns entirely (`remove_columns=["text","label"]`, `set_format(columns=["input_ids","attention_mask"])`); labels compared from the original dataframe afterward instead |
| 9 | `print()`-ing dataset text containing emoji crashed with `UnicodeEncodeError` | Windows console defaults to cp1252, not UTF-8 | Write such output to a file with explicit `encoding="utf-8"` instead of printing to console |
| 10 | Gemini judge silently discarded ~500 already-judged rows after hitting daily quota | `load_done_ids()` treated *any* `row_id` present in the output file as permanently done, including rows that had been recorded as a generic `error:` line from a caught `RESOURCE_EXHAUSTED` exception | Catch `RESOURCE_EXHAUSTED` specifically and `break` the loop immediately instead of writing an error row and continuing; manually stripped the 43 bad rows already written so they'd be retried |
| 11 | Same quota-discard bug recurred one day later | The fix from #10 was only applied via an explicit `--model` CLI flag in the resume command; the *default* model string in both `run_gemini_judge.py`'s argparse and `gemini_judge.py`'s `run_judge()` signature was still the old, broken model name, so a bare rerun (no flag) hit it again — 11 more rows wasted | Updated the default in both files to `gemini-flash-lite-latest` so the bare command works correctly without requiring an explicit flag |
| 12 | `pip install llm-guard` into the main venv downgraded core pinned packages | `llm-guard`'s dependency tree (spacy, presidio, nltk, faker, thinc, blis, etc.) forced `transformers`/`huggingface-hub`/`tokenizers`/`cryptography` down to older versions, risking silently reintroducing bug #2 | Uninstalled `llm-guard` and its exclusive transitive deps from the main venv, force-reinstalled the exact prior pinned versions, verified via `import transformers; transformers.__version__`; created a fully separate `venv-llmguard/` for all future LLM Guard work |
| 13 | One-off script at `/tmp/run_compare3.py` failed with `ModuleNotFoundError: No module named 'src'` | `/tmp` is outside the repo, so `src` wasn't importable relative to it (`sys.path[0]` was `/tmp`, not the repo root) | Wrote a proper `scripts/compare_three_way.py` inside the repo instead, consistent with the project's existing thin-entry-point convention |
| 14 | `run_demo.py`'s hardcoded `host="127.0.0.1"` would be unreachable from a Docker container even with `-p` port mapping | Binding to the loopback address only accepts connections originating *inside* the same network namespace, which a host machine reaching in via `-p` is not | Read `HOST`/`PORT` from environment variables (default unchanged for local use); Docker image sets `ENV HOST=0.0.0.0` |
| 15 | `.dockerignore`'s blanket `models/` entry hid the very files the Dockerfile needed to `COPY` | Ignored paths never reach the Docker build context at all, regardless of whether a later `COPY` instruction references them | Used explicit per-level re-inclusion (`!models/...`) for each parent directory down to the 4 needed files |
| 16 | `docker build` failed twice with `DeadlineExceeded` reaching `auth.docker.io` | (1) Docker Desktop's daemon wasn't running at all; (2) after starting it, a genuine mobile-hotspot network issue was selectively blocking `auth.docker.io`/`github.com` while other hosts worked fine | Started Docker Desktop and waited for `docker info` to succeed; diagnosed the network issue methodically (ruling out Docker itself, then DNS, then IPv4/IPv6) before concluding it was network-side; retried successfully once connectivity stabilized |
| 17 | A backgrounded `docker build ... \| tail -100` reported a misleadingly successful task-completion status even when the build had actually failed | The harness's completion-status tracking saw `tail`'s own exit code (0), not `docker build`'s real exit code, because the pipe hid it | Appended `; echo "EXIT_CODE:${PIPESTATUS[0]}"` to any piped/backgrounded command and checked that line explicitly instead of trusting the task-notification status alone |

## 12. Complete metrics appendix

All values below regenerated this session (2026-08-11) with the added `brier_score`/`ece`
(§5) and `cohen_kappa`/`pairwise_agreement` (§8) fields.

### DeBERTa, full test split (47,024 rows), threshold = 0.5 (raw model output)

| | overall | necent | neuralchemy | qualifire |
|---|---|---|---|---|
| n | 47,024 | 46,085 | 439 | 500 |
| accuracy | 0.99396 | 0.99666 | 0.95900 | 0.77600 |
| precision | 0.99405 | 0.99811 | 0.96255 | 0.64103 |
| recall | 0.99639 | 0.99661 | 0.96981 | 1.00000 |
| f1 | 0.99522 | 0.99736 | 0.96617 | 0.78125 |
| mcc | 0.98703 | 0.99281 | 0.91419 | 0.63381 |
| roc_auc | 0.99959 | 0.99975 | 0.98887 | 0.99007 |
| pr_auc | 0.99977 | 0.99986 | 0.99214 | 0.98668 |
| brier_score | 0.00570 | 0.00317 | 0.03885 | 0.21018 |
| ece | 0.00519 | 0.00267 | 0.03819 | 0.21844 |
| tp/tn/fp/fn | 29546/17194/177/107 | 29089/16842/55/99 | 257/164/10/8 | 200/188/112/0 |
| eval_loss | 0.02911 | — | — | — |
| latency mean/p50/p95/p99 (ms) | 54.85 / 48.41 / 101.93 / 155.32 | — | — | — |

### DeBERTa, full val split (47,023 rows), threshold = 0.5

| | overall | necent | neuralchemy | qualifire |
|---|---|---|---|---|
| n | 47,023 | 46,084 | 439 | 500 |
| accuracy | 0.99460 | 0.99733 | 0.96355 | 0.77000 |
| f1 | 0.99572 | 0.99789 | 0.97015 | 0.77756 |
| mcc | 0.98840 | 0.99426 | 0.92376 | 0.62564 |
| brier_score | 0.00504 | 0.00249 | 0.03453 | 0.21405 |
| ece | 0.00460 | 0.00199 | 0.03468 | 0.22217 |
| eval_loss | 0.02555 | — | — | — |

### Threshold sweep, test split — deployed threshold vs. alternatives

| threshold | overall F1 | necent F1 | neuralchemy F1 | qualifire F1 | qualifire precision |
|---|---|---|---|---|---|
| 0.5 (default) | 0.9952 | 0.9974 | 0.9662 | 0.7812 | 0.6410 |
| **0.99 (deployed)** | 0.9879 | 0.9887 | 0.9439 | **0.9279** | 0.8935 |
| 0.998 | 0.9791 | 0.9798 | 0.9286 | 0.9381 | 0.9681 |

0.99 adopted: 0.998 only wins on qualifire itself (+1 F1 point) but costs more everywhere else
(necent/neuralchemy/overall) — dominated, not worth adopting.

### Three-way comparison (1,439-row shared sample)

| | DeBERTa | Gemini (n=1433) | LLM Guard |
|---|---|---|---|
| accuracy | 0.90757 | 0.84508 | 0.71091 |
| precision | 0.86121 | 0.90299 | 0.71945 |
| recall | 0.98568 | 0.79396 | 0.75130 |
| f1 | 0.91925 | 0.84497 | 0.73503 |
| mcc | 0.82249 | 0.69716 | 0.41779 |
| roc_auc / pr_auc | 0.99104 / 0.99256 | — | — |
| brier_score / ece | 0.08683 / 0.08836 | — | — |
| tp/tn/fp/fn | 757/549/122/11 | 605/606/65/157 | 577/446/225/191 |

Per-source F1 / MCC:

| source | DeBERTa F1 / MCC | Gemini F1 / MCC | LLM Guard F1 / MCC |
|---|---|---|---|
| necent (n=500/499/500) | 0.9950 / 0.9875 | 0.8509 / 0.6880 | 0.6667 / **-0.0359** |
| neuralchemy (n=439/438/439) | 0.9662 / 0.9142 | 0.8607 / 0.7111 | **0.9002** / 0.7881 |
| qualifire (n=500/496/500) | 0.7813 / 0.6338 | 0.8180 / 0.6950 | 0.6480 / 0.4214 |

Gemini refusal counts: necent 1/500, neuralchemy 1/439, qualifire 4/500 (6 total).

### Pairwise Cohen's Kappa (1,439-row shared sample)

| Pair | Cohen's Kappa | Raw agreement | n |
|---|---|---|---|
| DeBERTa vs Gemini | 0.66834 | 0.83182 | 1433 |
| DeBERTa vs LLM Guard | 0.43669 | 0.72550 | 1439 |
| Gemini vs LLM Guard | 0.44449 | 0.72017 | 1433 |

### Necent leakage detection (2,000-row test sample vs. 368,672-row train)

| Method | Threshold | % of sample above threshold |
|---|---|---|
| Embedding cosine sim | ≥0.99 | 5.15% |
| Embedding cosine sim | ≥0.95 | 10.55% |
| Embedding cosine sim | ≥0.90 | 14.65% |
| Embedding cosine sim | ≥0.80 | 33.00% |
| MinHash Jaccard | ≥0.9 | 4.45% |
| MinHash Jaccard | ≥0.8 | 7.25% |
| MinHash Jaccard | ≥0.5 | 13.45% |
| MinHash Jaccard | 0 candidates (no lexical match at all) | 84.60% |

Leakage-adjusted DeBERTa performance on the same 2,000-row sample:

| Subset | n | accuracy | f1 | mcc |
|---|---|---|---|---|
| leaked (embedding sim ≥0.99) | 103 | 1.0000 | 1.0000 | 1.0000 |
| leaked (embedding sim ≥0.95) | 211 | 1.0000 | 1.0000 | 1.0000 |
| clean (embedding sim <0.95) | 1,789 (89.45%) | 0.99609 | 0.99669 | 0.99191 |

Clean-subset F1 (99.67%) is essentially identical to necent's originally reported full-test-set F1
(~99.7-99.8%) — leakage is real but too small a share of the data to meaningfully inflate the
headline number.

### Qualifire language contamination check

500 rows, 488 English / 12 non-English (2.4% non-English rate). 112 false positives, **0** of them
on non-English text (0.0% FP non-English rate) — language contamination ruled out as an explanation
for qualifire's precision problem.
