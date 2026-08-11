FROM python:3.11-slim

WORKDIR /app

# CPU-only torch build, installed separately from requirements-demo.txt so it
# never pulls in the CUDA build (see requirements-demo.txt for why).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY src/ src/
COPY scripts/run_demo.py scripts/run_demo.py
COPY demo/ demo/

# Only the files load_model() actually needs at inference time - excludes
# optimizer.pt/scheduler.pt/rng_state.pth (~1.4GB of training-only state that
# a checkpoint directory carries but inference never touches).
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
