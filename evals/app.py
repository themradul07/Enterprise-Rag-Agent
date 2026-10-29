# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import nest_asyncio
import pandas as pd
import streamlit as st

nest_asyncio.apply()

import importlib
import evals.metrics
import evals.storage
importlib.reload(evals.metrics)
importlib.reload(evals.storage)

from evals.pipeline import run_pipeline, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics
from evals.storage import (
    save_enriched_dataset,
    load_enriched_dataset,
    save_metric_results,
    load_metric_results,
    save_guardrails_results,
    load_guardrails_results,
    clear_saved_eval_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG — Eval Suite",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    "green":  "#d4edda",
    "yellow": "#fff3cd",
    "red":    "#f8d7da",
}


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"


def _color_score(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 0.75:
        return f"background-color: {SCORE_COLORS['green']}"
    elif val >= 0.5:
        return f"background-color: {SCORE_COLORS['yellow']}"
    return f"background-color: {SCORE_COLORS['red']}"


def _render_metric_table(df: pd.DataFrame, metric_col: str, title: str):
    avg = df[metric_col].mean()
    st.markdown(f"**{title}** — AVG: {_badge(avg)} `{avg:.2f}` {_grade(avg)}")
    styled = df.style.applymap(_color_score, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, width='stretch', hide_index=True)


def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init (with auto-retrieval from local storage)
# ─────────────────────────────────────────────────────────────────────────────
if "golden" not in st.session_state:
    st.session_state.golden = load_golden_dataset()

if "enriched_dataset" not in st.session_state:
    saved_enriched = load_enriched_dataset()
    if saved_enriched:
        st.session_state.enriched_dataset = saved_enriched
        st.session_state.pipeline_done = True
    else:
        st.session_state.enriched_dataset = None
        st.session_state.pipeline_done = False

if "guardrails_results" not in st.session_state:
    st.session_state.guardrails_results = load_guardrails_results()

if "metric_results" not in st.session_state:
    saved_metrics = load_metric_results()
    st.session_state.metric_results = saved_metrics if saved_metrics else {}

if "pipeline_rows" not in st.session_state:
    if st.session_state.enriched_dataset:
        st.session_state.pipeline_rows = [
            {
                "#": idx + 1,
                "Question": (s["question"][:55] + "...") if len(s["question"]) > 55 else s["question"],
                "Live Response (truncated)": (s.get("actual_response") or "")[:80] + "...",
                "Status": "✅" if s.get("actual_response") else "❌",
            }
            for idx, s in enumerate(st.session_state.enriched_dataset.get("rag_samples", []))
            if s.get("actual_response")
        ]
    else:
        st.session_state.pipeline_rows = []

golden = st.session_state.golden

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 Enterprise RAG — Evaluation Suite")
st.caption(
    "Step 1: Review ground truth → Step 2: Run live pipeline → Step 3: Score with RAGAS"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📋 Step 1 — Ground Truth", "🚀 Step 2 — Live Pipeline", "📊 Step 3 — Eval Metrics"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ground Truth Dataset")
    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer (ground truth), and the expected tool the RAG agent should call."
    )

    rag_rows = []
    for s in golden["rag_samples"]:
        rag_rows.append({
            "ID": s["id"],
            "Domain": s["domain"].replace("_", " ").title(),
            "Question": s["question"],
            "Reference Answer": s["reference"][:120] + "..." if len(s["reference"]) > 120 else s["reference"],
            "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—",
        })
    df_golden = pd.DataFrame(rag_rows)
    st.dataframe(df_golden, width='stretch', hide_index=True)
    st.caption(f"✅ {len(rag_rows)} golden RAG samples from 5 enterprise docs")

    st.divider()

    st.subheader("Guardrails Test Cases")
    st.markdown(
        "These inputs test whether the safety rails correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []
    for g in golden["guardrails_samples"]:
        expected_label = "🛡️ Block" if g["expected_blocked"] else "✅ Pass"
        g_rows.append({
            "ID": g["id"],
            "Input": g["input"],
            "Expected": expected_label,
            "Type": g["type"],
            "Description": g["description"],
        })
    st.dataframe(pd.DataFrame(g_rows), width='stretch', hide_index=True)
    st.caption("6 guardrails test cases: 3 adversarial (should block) + 3 legit (should pass)")

    with st.expander("View raw golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
    st.markdown(
        "Sends each golden question to your **running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Responses are truncated to 300 chars to save tokens for the RAGAS judging step."
    )
    st.info(
        "⚠️ Make sure your FastAPI backend is running first: `uvicorn app.main:app --reload --port 8000`",
        icon="⚠️",
    )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    run_pipeline_btn = col_p1.button(
        "▶️ Run Live Pipeline",
        type="primary",
        width="stretch",
        disabled=st.session_state.pipeline_done,
    )
    reset_btn = col_p2.button(
        "🔄 Reset & Re-run",
        width="stretch",
        disabled=not st.session_state.pipeline_done,
    )

    if reset_btn:
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = {}
        st.session_state.pipeline_rows = []
        clear_saved_eval_data()
        st.rerun()

    if run_pipeline_btn:
        st.session_state.pipeline_rows = []
        progress_bar = st.progress(0, text="Starting pipeline...")
        live_table_slot = st.empty()
        status_slot = st.empty()

        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
            if stage == "calling":
                progress_bar.progress(pct, text=f"[{i+1}/{total}] Calling /query: {question[:60]}...")
            else:
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
                st.session_state.pipeline_rows.append({
                    "#": i + 1,
                    "Question": short_q,
                    "Live Response (truncated)": short_r if short_r else "⚠️ No response",
                    "Status": "✅" if short_r else "❌",
                })
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    width='stretch',
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i+1}/{total}] ✅ Done",
                )

        with logfire.span("🚀 Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(golden, progress_callback=pipeline_cb)
            st.session_state.enriched_dataset = enriched
            save_enriched_dataset(enriched)

        progress_bar.progress(100, text="✅ All responses collected!")
        status_slot.success(f"💾 {len(enriched['rag_samples'])} responses stored in session and saved to disk.")

        # ── Guardrails tests ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Guardrails Tests")
        g_progress = st.progress(0, text="Running guardrails tests...")
        g_status_slot = st.empty()

        def g_cb(i, total, input_text):
            g_progress.progress(
                int((i / total) * 100),
                text=f"[{i+1}/{total}] Testing: {input_text[:60]}...",
            )

        with logfire.span("🛡️ Streamlit — Guardrails Tests"):
            g_results = run_guardrails_eval(enriched["guardrails_samples"], progress_callback=g_cb)
            g_metrics = compute_guardrails_metrics(g_results)
            st.session_state.guardrails_results = g_results
            save_guardrails_results(g_results)
            st.session_state.pipeline_done = True

        g_progress.progress(100, text="✅ Guardrails tests complete!")

        g_rows_live = []
        for r in g_results:
            result_label = {
                "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                "FP": "🛡️ Blocked ❌ (False Positive)", "FN": "✅ Passed ❌ (Missed)",
            }.get(r["result"], r["result"])
            g_rows_live.append({
                "ID": r["id"],
                "Input": r["input"][:70],
                "Expected": "🛡️ Block" if r["expected_blocked"] else "✅ Pass",
                "Actual": "Blocked" if r["actual_blocked"] else "Passed",
                "Result": result_label,
            })
        st.dataframe(pd.DataFrame(g_rows_live), width='stretch', hide_index=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['total']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")

    elif st.session_state.pipeline_done:
        st.success("✅ Pipeline already run. See results below.")

        resp_rows = []
        for s in st.session_state.enriched_dataset["rag_samples"]:
            resp_rows.append({
                "#": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"][:60],
                "Live Response": s["actual_response"][:100] + "..." if len(s.get("actual_response","")) > 100 else s.get("actual_response",""),
                "Tool Called": s["actual_tools_called"][0] if s.get("actual_tools_called") else "—",
                "Contexts Retrieved": len(s.get("actual_contexts", [])),
            })
        st.dataframe(pd.DataFrame(resp_rows), width='stretch', hide_index=True)

        if st.session_state.guardrails_results:
            st.divider()
            st.subheader("Guardrails Results (from previous run)")
            g_rows_prev = []
            for r in st.session_state.guardrails_results:
                result_label = {
                    "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                    "FP": "Blocked ❌ FP", "FN": "Passed ❌ FN",
                }.get(r["result"], r["result"])
                g_rows_prev.append({
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Result": result_label,
                })
            st.dataframe(pd.DataFrame(g_rows_prev), width='stretch', hide_index=True)
            gm = compute_guardrails_metrics(st.session_state.guardrails_results)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Correct", f"{gm['correct']}/{gm['total']}")
            mc2.metric("Precision", f"{gm['precision']:.2f}")
            mc3.metric("Recall", f"{gm['recall']:.2f}")
            mc4.metric("Accuracy", f"{gm['accuracy']:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")

    if not st.session_state.pipeline_done or not st.session_state.enriched_dataset:
        st.warning("⚠️ Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        metric_display_names = {
            "faithfulness":      "Exp 1 — Faithfulness",
            "answer_relevancy":  "Exp 2 — Answer Relevancy",
            "context_precision": "Exp 3 — Context Precision",
            "context_recall":    "Exp 4 — Context Recall",
            "answer_correctness":"Exp 5 — Answer Correctness",
            "tool_correctness":  "Exp 6 — Tool Correctness",
        }

        valid_samples = [
            s for s in st.session_state.enriched_dataset.get("rag_samples", [])
            if (s.get("actual_response") or "").strip()
        ]
        total_available = len(valid_samples)

        # ── Controls row ──────────────────────────────────────────────────────
        col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 2])
        with col_cfg1:
            sample_limit = st.slider(
                "Samples to Evaluate",
                min_value=1,
                max_value=max(1, total_available),
                value=min(3, total_available),
                help="Evaluate a small subset (e.g. 3-5 samples) for quick iteration (1-3 min) without hitting rate limits.",
            )
        with col_cfg2:
            cooldown_choice = st.select_slider(
                "Cooldown between samples",
                options=[10, 15, 20, 30, 40],
                value=15,
                format_func=lambda x: f"{x}s ({'Fast' if x <= 15 else 'Conservative'})",
                help="Cooldown between samples. 15s is safe with your dedicated JUDGE_GROQ key and truncated context.",
            )
        with col_cfg3:
            selected_metrics = st.multiselect(
                "Select Experiments to Run",
                options=list(metric_display_names.keys()),
                default=list(metric_display_names.keys()),
                format_func=lambda k: metric_display_names[k],
                help="Select which metrics to score. Uncheck to skip and save time.",
            )

        # ── Runtime estimate banner ───────────────────────────────────────────
        llm_count = len([m for m in selected_metrics if m != "tool_correctness"])
        est_sec = (sample_limit * cooldown_choice * max(0, llm_count)) + (llm_count * 20)
        est_min = max(0.2, round(est_sec / 60, 1))

        has_judge_key = bool(os.getenv("JUDGE_GROQ"))
        key_label = "`JUDGE_GROQ` (dedicated eval key)" if has_judge_key else "`GROQ_API_KEY`"
        st.info(
            f"⚡ Key used: **{key_label}** | "
            f"⏱️ Estimated runtime for **{sample_limit} sample(s)** across **{len(selected_metrics)} metric(s)**: **~{est_min} min(s)**. "
            f"💾 Results are automatically saved to disk (`evals/results/metric_results.json`) as each experiment completes.",
            icon="ℹ️",
        )

        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        run_metrics_btn = btn_col1.button(
            "▶️ Run Selected Metrics",
            type="primary",
            disabled=not st.session_state.pipeline_done,
        )
        clear_btn = btn_col2.button("🗑️ Clear Saved Metrics")

        if clear_btn:
            st.session_state.metric_results = {}
            save_metric_results({})
            st.success("Cleared saved metrics from local storage.")
            st.rerun()

        if run_metrics_btn:
            if not selected_metrics:
                st.warning("⚠️ Please select at least one metric to evaluate.")
                st.stop()

            status_slot = st.empty()
            results_slots = {key: st.empty() for key in metric_display_names.keys()}

            # Pre-render already computed metrics
            for key in metric_display_names.keys():
                if key in st.session_state.metric_results:
                    with results_slots[key].container():
                        _render_metric_table(st.session_state.metric_results[key], key, metric_display_names[key])

            def status_cb(msg: str):
                status_slot.info(msg)

            def metric_done_cb(key: str, df: pd.DataFrame):
                st.session_state.metric_results[key] = df
                save_metric_results(st.session_state.metric_results)
                if key in results_slots:
                    with results_slots[key].container():
                        _render_metric_table(df, key, metric_display_names[key])

            with logfire.span("📊 Streamlit — Run Metrics Button"):
                try:
                    metric_results = _run_async(
                        run_all_metrics(
                            st.session_state.enriched_dataset,
                            status_cb=status_cb,
                            metric_cb=metric_done_cb,
                            selected_metrics=selected_metrics,
                            max_samples=sample_limit,
                            cooldown_mini=cooldown_choice,
                            cooldown_standard=20,
                        )
                    )
                    st.session_state.metric_results.update(metric_results)
                    save_metric_results(st.session_state.metric_results)
                    status_slot.success("✅ Evaluation complete! All results saved to local storage (`evals/results/metric_results.json`).")
                except Exception as e:
                    status_slot.error(f"❌ Metrics evaluation failed: {e}")
                    logfire.error("Metrics evaluation failed", error=str(e))
                    st.stop()

        elif st.session_state.metric_results:
            st.success("💾 **Results loaded from local disk storage** (`evals/results/metric_results.json`).")
            for key, title in metric_display_names.items():
                if key in st.session_state.metric_results:
                    _render_metric_table(st.session_state.metric_results[key], key, title)

        # ── Final Summary ─────────────────────────────────────────────────────
        if st.session_state.metric_results:
            st.divider()
            st.subheader("Final Summary")

            mr = st.session_state.metric_results
            summary = [
                ("Faithfulness",       mr.get("faithfulness",      pd.DataFrame()).get("faithfulness",      pd.Series()).mean()),
                ("Answer Relevancy",   mr.get("answer_relevancy",  pd.DataFrame()).get("answer_relevancy",  pd.Series()).mean()),
                ("Context Precision",  mr.get("context_precision", pd.DataFrame()).get("context_precision", pd.Series()).mean()),
                ("Context Recall",     mr.get("context_recall",    pd.DataFrame()).get("context_recall",    pd.Series()).mean()),
                ("Answer Correctness", mr.get("answer_correctness",pd.DataFrame()).get("answer_correctness",pd.Series()).mean()),
                ("Tool Correctness",   mr.get("tool_correctness",  pd.DataFrame()).get("tool_correctness",  pd.Series()).mean()),
            ]

            cols = st.columns(len(summary))
            for col, (name, score) in zip(cols, summary):
                if pd.notna(score):
                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(score),
                    )

            if st.session_state.guardrails_results:
                gm = compute_guardrails_metrics(st.session_state.guardrails_results)
                st.metric(
                    label="🛡️ Guardrails Accuracy",
                    value=f"{gm['correct']}/{gm['total']}",
                    delta=f"Precision {gm['precision']:.2f} | Recall {gm['recall']:.2f}",
                )

            summary_df = pd.DataFrame([
                {"Metric": name, "Score": f"{score:.3f}" if pd.notna(score) else "—", "Grade": _grade(score) if pd.notna(score) else "—"}
                for name, score in summary
            ])
            st.dataframe(summary_df, width='stretch', hide_index=True)
