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

## Status

Work in progress — dataset acquisition and exploration phase.
