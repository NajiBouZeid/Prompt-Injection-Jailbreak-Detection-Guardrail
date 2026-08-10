import time

import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference.guardrail import Guardrail

app = FastAPI()

# Loaded once at process startup, not per-request - keeps /api/classify to a single
# forward pass instead of re-loading the checkpoint every time.
guardrail = Guardrail()

# Held out during training - safe to sample from for demo purposes without the
# model having ever seen these exact rows.
test_df = pd.read_parquet("data/processed/test.parquet")


class ClassifyRequest(BaseModel):
    text: str


@app.post("/api/classify")
def classify(request: ClassifyRequest):
    start = time.perf_counter()
    result = guardrail.classify(request.text)
    result["latency_ms"] = (time.perf_counter() - start) * 1000
    return result


@app.get("/api/sample")
def sample():
    row = test_df.sample(1).iloc[0]
    return {
        "text": row["text"],
        "true_label": "attack" if row["label"] == 1 else "benign",
        "source_dataset": row["source_dataset"],
    }


# Mounted last: FastAPI matches routes in registration order, so /api/classify
# above still wins over the catch-all static file server below.
app.mount("/", StaticFiles(directory="demo/static", html=True), name="static")
