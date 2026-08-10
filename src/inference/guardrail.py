import numpy as np
import torch

from src.evaluation.evaluate_classifier import MAX_LENGTH, load_model

DEFAULT_MODEL_DIR = "models/deberta-v3-base-guardrail/checkpoint-23512"

# Adopted 2026-08-10 via the global threshold sweep (src/evaluation/threshold_sweep.py):
# raises qualifire precision from 64% to 89% at the cost of a little necent/neuralchemy
# recall, confirmed on the held-out test split. See the eval report for the full tradeoff.
DEFAULT_THRESHOLD = 0.99


class Guardrail:
    """Loads the fine-tuned DeBERTa checkpoint once and classifies prompts against it.

    Kept as a class (not a bare function) so the model/tokenizer load - the slow part -
    happens once at construction, and classify() stays cheap for repeated calls, e.g.
    one request at a time behind an API.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, threshold: float = DEFAULT_THRESHOLD):
        self.model, self.tokenizer, self.device = load_model(model_dir)
        self.threshold = threshold

    def classify(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()[0]

        exp_logits = np.exp(logits - logits.max())
        prob_attack = float((exp_logits / exp_logits.sum())[1])

        return {
            "label": "attack" if prob_attack >= self.threshold else "benign",
            "prob_attack": prob_attack,
            "threshold": self.threshold,
        }
