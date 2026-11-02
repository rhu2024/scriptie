from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import importlib

from utils.openai_client import get_client
from utils.metrics import mean_pairwise_cosine
from utils.io import write_json
from utils.steps import extract_steps


# ==========================================================
# Data models
# ==========================================================
@dataclass
class RunResult:
    run: int
    answer: str
    latency_seconds: float
    usage: Dict[str, Any] | None


@dataclass
class SummaryResult:
    stability_S: float
    avg_latency_seconds: float
    avg_prompt_tokens: float | None
    avg_completion_tokens: float | None
    avg_total_tokens: float | None


# ==========================================================
# Helpers
# ==========================================================
def _usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """
    openai-python geeft 'usage' vaak als object terug. We maken het json-serializable.
    """
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    out = {}
    for k in dir(usage):
        if k.startswith("_"):
            continue
        try:
            v = getattr(usage, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out or None


def _avg_floats(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _avg_ints(xs: List[int]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _load_prompts(reload_module: bool = False):
    """
    Laadt configs.prompts dynamisch.
    reload_module=True helpt in notebooks om cache issues te voorkomen.
    """
    import configs.prompts as prompts  # lokale import -> volgt actuele sys.path
    if reload_module:
        prompts = importlib.reload(prompts)
    return prompts


def _get_prompt_bundle(*, context: Optional[str], reload_prompts: bool = False) -> Tuple[str, str, str, List[Dict[str, str]]]:
    """
    Returns:
      prompt_id, question, prompts_file_path, messages
    """
    prompts = _load_prompts(reload_module=reload_prompts)

    prompt_id = getattr(prompts, "PROMPT_ID", "UNKNOWN_PROMPT_ID")
    question = getattr(prompts, "QUESTION", "UNKNOWN_QUESTION")
    prompts_file = getattr(prompts, "__file__", "UNKNOWN_PROMPTS_FILE")

    if not hasattr(prompts, "build_messages"):
        raise RuntimeError(f"configs.prompts heeft geen build_messages(). File: {prompts_file}")

    # ✅ vraag zit in prompts.py, dus build_messages(context=...)
    messages = prompts.build_messages(context=context)

    return prompt_id, question, prompts_file, messages


# ==========================================================
# LLM calls
# ==========================================================
def chat_once(
    *,
    model: str,
    context: Optional[str],
    params: Dict[str, Any],
    reload_prompts: bool = False,
) -> Tuple[str, float, Optional[Dict[str, Any]], str, str, List[Dict[str, str]], str]:
    """
    1 chat completion call met centrale prompt (configs/prompts.py).

    Returns:
      answer,
      latency_seconds,
      usage(dict|None),
      prompt_id,
      question (uit prompts.py),
      messages (exact verzonden),
      prompts_file (pad naar configs/prompts.py die gebruikt is)
    """
    client = get_client()

    prompt_id, question, prompts_file, messages = _get_prompt_bundle(
        context=context,
        reload_prompts=reload_prompts,
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **params,
    )
    latency = time.time() - t0

    answer = resp.choices[0].message.content
    usage = _usage_to_dict(getattr(resp, "usage", None))
    return answer, latency, usage, prompt_id, question, messages, prompts_file


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Embeddings voor stability metric S.
    """
    client = get_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


# ==========================================================
# Legacy runner (per-case folders) - optional
# ==========================================================
def run_experiment_for_case(
    *,
    experiment_name: str,
    case_name: str,
    model: str,
    context: str,
    runs: int,
    chat_params: Dict[str, Any],
    results_base_dir: str,
    embedding_model: str = "text-embedding-3-small",
    logger: Optional[logging.Logger] = None,
    reload_prompts_each_call: bool = False,
) -> SummaryResult:
    """
    Legacy structuur:
    data/results_data/experiment_<exp>/case_<case>/run_*.json
    """
    logger = logger or logging.getLogger(__name__)

    case_dir = os.path.join(results_base_dir, f"experiment_{experiment_name}", f"case_{case_name}")
    os.makedirs(case_dir, exist_ok=True)

    answers: List[str] = []
    latencies: List[float] = []
    prompt_tokens: List[int] = []
    completion_tokens: List[int] = []
    total_tokens: List[int] = []

    used_prompt_id: Optional[str] = None
    used_question: Optional[str] = None
    used_prompts_file: Optional[str] = None

    for r in range(1, runs + 1):
        logger.info("Run %s/%s | exp=%s | case=%s", r, runs, experiment_name, case_name)

        answer, latency, usage, prompt_id, question, messages, prompts_file = chat_once(
            model=model,
            context=context,
            params=chat_params,
            reload_prompts=reload_prompts_each_call,
        )

        used_prompt_id = prompt_id
        used_question = question
        used_prompts_file = prompts_file

        answers.append(answer)
        latencies.append(latency)

        if usage:
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            if isinstance(pt, int):
                prompt_tokens.append(pt)
            if isinstance(ct, int):
                completion_tokens.append(ct)
            if isinstance(tt, int):
                total_tokens.append(tt)

        run_payload = {
            "run": r,
            "experiment": experiment_name,
            "case_id": case_name,
            "model": model,
            "prompt_id": prompt_id,
            "question": question,
            "prompts_file": prompts_file,
            "chat_params": chat_params,
            "messages": messages,
            "answer": answer,
            "latency_seconds": latency,
            "usage": usage,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(os.path.join(case_dir, f"run_{r}.json"), run_payload)

    embeddings = embed_texts(answers, model=embedding_model)
    stability = mean_pairwise_cosine(embeddings)

    summary = {
        "experiment": experiment_name,
        "case_id": case_name,
        "model": model,
        "prompt_id": used_prompt_id,
        "question": used_question,
        "prompts_file": used_prompts_file,
        "runs": runs,
        "embedding_model": embedding_model,
        "stability_S": stability,
        "avg_latency_seconds": _avg_floats(latencies),
        "avg_prompt_tokens": _avg_ints(prompt_tokens),
        "avg_completion_tokens": _avg_ints(completion_tokens),
        "avg_total_tokens": _avg_ints(total_tokens),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(os.path.join(case_dir, "summary.json"), summary)

    return SummaryResult(
        stability_S=stability,
        avg_latency_seconds=float(_avg_floats(latencies) or 0.0),
        avg_prompt_tokens=_avg_ints(prompt_tokens),
        avg_completion_tokens=_avg_ints(completion_tokens),
        avg_total_tokens=_avg_ints(total_tokens),
    )


# ==========================================================
# NEW runner: per experiment folder, per RUN file with ALL cases
# ==========================================================
def run_experiment_aggregated_runs(
    *,
    experiment_name: str,
    model: str,
    runs: int,
    chat_params: Dict[str, Any],
    cases: List[Dict[str, Any]],
    results_base_dir: str,
    embedding_model: str = "text-embedding-3-small",
    logger: Optional[logging.Logger] = None,
    reload_prompts_each_call: bool = False,
) -> Dict[str, Any]:
    """
    Structuur:

    data/results_data/
      experiment_<experiment_name>/
        run_1.json      # bevat ALLE cases
        run_2.json
        ...
        summary.json

    cases input:
      [
        {"case_id":"case_1", "context":"...", "metadata": {...}},
        ...
      ]
    """
    logger = logger or logging.getLogger(__name__)

    exp_dir = os.path.join(results_base_dir, f"experiment_{experiment_name}")
    os.makedirs(exp_dir, exist_ok=True)

    answers_by_case: Dict[str, List[str]] = {c["case_id"]: [] for c in cases}

    latencies_all: List[float] = []
    prompt_tokens_all: List[int] = []
    completion_tokens_all: List[int] = []
    total_tokens_all: List[int] = []

    first_prompt_id: Optional[str] = None
    first_question: Optional[str] = None
    first_prompts_file: Optional[str] = None

    for r in range(1, runs + 1):
        logger.info("===== RUN %s/%s | exp=%s =====", r, runs, experiment_name)

        run_cases_payload: List[Dict[str, Any]] = []
        run_timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for case in cases:
            case_id = case["case_id"]
            context = case.get("context", "")

            logger.info("Run %s | case=%s", r, case_id)

            answer, latency, usage, prompt_id, question, messages, prompts_file = chat_once(
                model=model,
                context=context,
                params=chat_params,
                reload_prompts=reload_prompts_each_call,
            )

            if first_prompt_id is None:
                first_prompt_id = prompt_id
            if first_question is None:
                first_question = question
            if first_prompts_file is None:
                first_prompts_file = prompts_file

            answers_by_case[case_id].append(answer)
            latencies_all.append(latency)

            if usage:
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                if isinstance(pt, int):
                    prompt_tokens_all.append(pt)
                if isinstance(ct, int):
                    completion_tokens_all.append(ct)
                if isinstance(tt, int):
                    total_tokens_all.append(tt)

            # Extraheer CoT-stappen voor individuele analyse
            steps = extract_steps(answer) if answer else {}

            run_cases_payload.append(
                {
                    "case_id": case_id,
                    "metadata": case.get("metadata", {}),
                    "prompt_id": prompt_id,
                    "question": question,          # ✅ uit prompts.py
                    "prompts_file": prompts_file,
                    "messages": messages,          # ✅ bewijs van prompt
                    "answer": answer,
                    "steps": steps,                # ✅ individuele CoT-stappen
                    "latency_seconds": latency,
                    "usage": usage,
                }
            )

        run_payload = {
            "run": r,
            "experiment": experiment_name,
            "model": model,
            "chat_params": chat_params,
            "timestamp_utc": run_timestamp_utc,
            "cases": run_cases_payload,
        }
        write_json(os.path.join(exp_dir, f"run_{r}.json"), run_payload)

    # Stability per case (over runs)
    stability_per_case: Dict[str, float] = {}
    for case_id, answers in answers_by_case.items():
        embs = embed_texts(answers, model=embedding_model)
        stability_per_case[case_id] = mean_pairwise_cosine(embs)

    overall_stability = (
        sum(stability_per_case.values()) / len(stability_per_case)
        if stability_per_case
        else None
    )

    summary = {
        "experiment": experiment_name,
        "model": model,
        "prompt_id": first_prompt_id,
        "question": first_question,
        "prompts_file": first_prompts_file,
        "runs": runs,
        "num_cases": len(cases),
        "embedding_model": embedding_model,
        "stability_per_case_S": stability_per_case,
        "overall_stability_S": overall_stability,
        "avg_latency_seconds": _avg_floats(latencies_all),
        "avg_prompt_tokens": _avg_ints(prompt_tokens_all),
        "avg_completion_tokens": _avg_ints(completion_tokens_all),
        "avg_total_tokens": _avg_ints(total_tokens_all),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    write_json(os.path.join(exp_dir, "summary.json"), summary)
    return summary