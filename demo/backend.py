import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference.guardrail import Guardrail

app = FastAPI()

# Loaded once at process startup, not per-request - keeps /api/classify to a single
# forward pass instead of re-loading the checkpoint every time.
guardrail = Guardrail()


class ClassifyRequest(BaseModel):
    text: str


@app.post("/api/classify")
def classify(request: ClassifyRequest):
    start = time.perf_counter()
    result = guardrail.classify(request.text)
    result["latency_ms"] = (time.perf_counter() - start) * 1000
    return result


# Mounted last: FastAPI matches routes in registration order, so /api/classify
# above still wins over the catch-all static file server below.
app.mount("/", StaticFiles(directory="demo/static", html=True), name="static")
