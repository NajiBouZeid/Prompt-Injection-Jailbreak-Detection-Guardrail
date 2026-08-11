# Prompt Injection & Jailbreak Detection Guardrail

A real-time guardrail that detects prompt injection and jailbreak attempts before they reach a target LLM.

Two detection approaches are built and benchmarked against each other:
- **Fine-tuned classifier** — a DeBERTa encoder fine-tuned to label a prompt as `attack` or `safe`.
- **LLM-as-judge** — a general-purpose LLM prompted to make the same judgment, no training required.

Both are evaluated on the same labeled dataset and compared on precision, recall, and latency, to understand the real tradeoff between a small dedicated classifier and a general-purpose LLM used as a filter. This project relates to **LLM01: Prompt Injection** in the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. **Important (Windows/NVIDIA GPU users):** the command above installs the CPU-only build of PyTorch. To use your GPU, reinstall torch with the CUDA-enabled build:
   ```
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   Verify it worked:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   This should print `True`.

## Demo

A local web demo serves the fine-tuned checkpoint behind a small FastAPI backend with a
browser frontend — paste a prompt, get an attack/benign verdict with the model's confidence.

```
python -m scripts.run_demo
```

Then open http://127.0.0.1:8000 in a browser. Everything runs locally; no prompt text is sent
anywhere external. Full training/evaluation results, including how the deployed decision
threshold was chosen, are in the [training & evaluation report](https://claude.ai/code/artifact/671c0fc4-aa8d-4287-a3ff-28c83aa8cd13).

For scripted/CLI use instead of the web UI:
```
python -m scripts.classify "some prompt to check"
```

### Running the demo in Docker

The demo can also run in a container, so it works on any machine with Docker installed and
no manual Python/venv setup — useful for reproducing the exact environment elsewhere.

```
docker build -t prompt-guardrail-demo .
docker run -p 8000:8000 prompt-guardrail-demo
```

Then open http://localhost:8000, same as the local (non-Docker) demo above.

The image is CPU-only (no GPU passthrough required) and installs from `requirements-demo.txt`
rather than the full `requirements.txt` — it only needs the packages the demo actually imports
at runtime, not the training/evaluation-only ones (`sentence-transformers`, `google-genai`,
`datasketch`, `langdetect`, `accelerate`). It also only copies the four checkpoint files
`load_model()` actually reads (`config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`), not the full `models/deberta-v3-base-guardrail/checkpoint-23512/`
directory — the rest of that directory is training-only state (optimizer/scheduler/RNG),
about 1.4GB the container never touches.

## Status

DeBERTa classifier trained and evaluated (99.4% test accuracy, see the report above for the
full breakdown including known weak spots and how they were addressed). Benchmarked against
an LLM-as-judge (Gemini) and the LLM Guard baseline. Local demo working end-to-end.
