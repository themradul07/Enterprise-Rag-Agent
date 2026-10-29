"""
Local persistence for evaluation datasets and metric results.
Saves results into evals/results/ so data is never lost across browser
refreshes, crashes, or server restarts.
"""

import os
import json
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ENRICHED_FILE = os.path.join(RESULTS_DIR, "enriched_dataset.json")
METRICS_FILE = os.path.join(RESULTS_DIR, "metric_results.json")
GUARDRAILS_FILE = os.path.join(RESULTS_DIR, "guardrails_results.json")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_enriched_dataset(data: dict):
    if not data:
        return
    _ensure_dir()
    with open(ENRICHED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_enriched_dataset() -> dict | None:
    if os.path.exists(ENRICHED_FILE):
        try:
            with open(ENRICHED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and isinstance(data, dict) and data.get("rag_samples"):
                    return data
        except Exception:
            return None
    return None


def save_metric_results(results: dict):
    if not results:
        return
    _ensure_dir()
    # Read existing on disk to merge so we never overwrite other metrics
    existing = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            existing[k] = v.to_dict(orient="records")
        elif isinstance(v, list):
            existing[k] = v

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def load_metric_results() -> dict:
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            parsed = {}
            for k, records in raw.items():
                if isinstance(records, list) and len(records) > 0:
                    parsed[k] = pd.DataFrame(records)
            return parsed
        except Exception:
            return {}
    return {}


def save_guardrails_results(results: list):
    if not results:
        return
    _ensure_dir()
    with open(GUARDRAILS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_guardrails_results() -> list | None:
    if os.path.exists(GUARDRAILS_FILE):
        try:
            with open(GUARDRAILS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def clear_saved_eval_data():
    for f in [ENRICHED_FILE, METRICS_FILE, GUARDRAILS_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass
