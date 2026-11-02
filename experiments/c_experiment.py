"""
Experiment C: Baseline + Internet Search

Opslagstructuur:

data/results_data/
  experiment_c/
    run_1.json
    run_2.json
    ...
    summary.json

Web cache:
data/web_cache/

Gebruik:
from experiments.c_experiment import run
run(project_root="/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu")
"""

from __future__ import annotations

import os
import sys
import glob
import time
import logging
import subprocess
from typing import Dict, Any, Optional, List


# ==========================================================
# Dependency installer (serverless compute heeft geen packages)
# ==========================================================
def _ensure_deps() -> None:
    """Installeer benodigde packages als ze ontbreken (voor serverless)."""
    required = ["openai", "requests", "bs4", "sentence_transformers"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        # Map import-namen naar pip-namen
        pip_map = {"bs4": "beautifulsoup4", "sentence_transformers": "sentence-transformers"}
        pip_pkgs = [pip_map.get(p, p) for p in missing]
        print(f"Installing missing packages: {pip_pkgs}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + pip_pkgs
        )


# =========================
# EXPERIMENT SETTINGS (C)
# =========================
EXPERIMENT: Dict[str, Any] = {
    "name": "c",
    "model": "gpt-4o-mini",
    "runs": 5,
    "chat_params": {
        "temperature": 0.8,
        "top_p": 1.0,
        "max_tokens": 10000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "web_search": {
        "backend": "openai",           # "openai" (primary) of "duckduckgo" (legacy)
        "model": "gpt-4o-mini",         # model voor OpenAI web search
        "max_results": 8,
        "fetch_top_n": 5,
        "use_cache": True,              # cache resultaten om API-kosten te beperken
        "request_sleep_seconds": 0.5,   # pauze tussen cases (rate limiting)
        # Legacy DuckDuckGo instellingen (alleen bij backend="duckduckgo")
        "query_suffix": "best practices",
    },
}


# ==========================================================
# Project root helpers
# ==========================================================
def _is_project_root(path: str) -> bool:
    return (
        os.path.isdir(os.path.join(path, "utils"))
        and os.path.isdir(os.path.join(path, "configs"))
        and os.path.isdir(os.path.join(path, "experiments"))
        and os.path.isdir(os.path.join(path, "data"))
    )


def _sanitize_syspath(project_root: str) -> None:
    project_root = os.path.abspath(project_root)

    new_path = []
    for p in sys.path:
        try:
            ap = os.path.abspath(p)
        except Exception:
            ap = p

        if "AI_op_het_menu" in str(p) and ap != project_root:
            continue
        new_path.append(p)

    sys.path[:] = new_path

    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)


def _setup_logger(log_path: str) -> logging.Logger:
    logger_name = f"experiment_c_logger_{os.path.abspath(log_path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Logger geïnitialiseerd. Schrijf naar logbestand: %s", log_path)

    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    return logger


# ==========================================================
# Main
# ==========================================================
def run(
    project_root: str,
    cases_glob: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    clean_output: bool = True,
) -> str:
    """
    Draait experiment C:
    - haalt per case aanvullende internetinformatie op via web search
    - voegt die toe aan de context
    - draait daarna dezelfde runner als experiment A/B

    Ondersteunt twee search backends:
    - "openai": OpenAI Responses API met web_search_preview (aanbevolen)
    - "duckduckgo": Legacy DuckDuckGo scraping (werkt NIET vanuit serverless)

    Output:
    <PROJECT_ROOT>/data/results_data/experiment_c/run_*.json
    """
    # Installeer packages als ze ontbreken (serverless compute)
    _ensure_deps()

    if not project_root:
        raise ValueError("project_root is verplicht.")

    project_root = os.path.abspath(project_root)
    if not _is_project_root(project_root):
        raise ValueError(f"project_root lijkt geen geldige projectmap: {project_root}")

    _sanitize_syspath(project_root)

    from utils.io import read_case_file
    from utils.runner import run_experiment_aggregated_runs
    from utils.web_search import (
        build_search_query,
        retrieve_web_context,
        retrieve_web_context_openai,
        build_augmented_web_context,
    )

    import configs.prompts as prompts
    import importlib
    prompts = importlib.reload(prompts)

    case_dir = os.path.join(project_root, "data", "case_data")
    web_cache_dir = os.path.join(project_root, "data", "web_cache")
    results_dir = os.path.join(project_root, "data", "results_data")
    logs_dir = os.path.join(project_root, "logs")

    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(web_cache_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    exp_name = EXPERIMENT["name"]
    model = EXPERIMENT["model"]
    runs = int(EXPERIMENT["runs"])
    chat_params = dict(EXPERIMENT["chat_params"])

    web_cfg = dict(EXPERIMENT["web_search"])
    search_backend = str(web_cfg.get("backend", "openai"))
    search_model = str(web_cfg.get("model", "gpt-4o-mini"))
    max_results = int(web_cfg["max_results"])
    fetch_top_n = int(web_cfg["fetch_top_n"])
    use_cache = bool(web_cfg.get("use_cache", True))
    request_sleep_seconds = float(web_cfg.get("request_sleep_seconds", 0.0))
    query_suffix = str(web_cfg.get("query_suffix", "")).strip()

    import shutil
    exp_output_dir = os.path.join(results_dir, f"experiment_{exp_name}")
    if clean_output and os.path.exists(exp_output_dir):
        shutil.rmtree(exp_output_dir)
    os.makedirs(exp_output_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, f"experiment_{exp_name}.log")
    logger = _setup_logger(log_path)

    if cases_glob is None:
        cases_glob = os.path.join(case_dir, "case_*.*")

    case_paths = sorted(glob.glob(cases_glob))
    if not case_paths:
        raise ValueError(f"Geen cases gevonden met glob: {cases_glob}")

    logger.info("==== START experiment_%s ====", exp_name)
    logger.info("Project root: %s", project_root)
    logger.info("Using prompts from: %s", prompts.__file__)
    logger.info("PROMPT_ID: %s", getattr(prompts, "PROMPT_ID", "<missing PROMPT_ID>"))
    logger.info("QUESTION: %s", getattr(prompts, "QUESTION", "<missing QUESTION in prompts.py>"))
    logger.info("Cases: %s", len(case_paths))
    logger.info("Web search backend: %s", search_backend)
    logger.info("Web cache dir: %s", web_cache_dir)
    logger.info("Output dir: %s", exp_output_dir)
    logger.info(
        "Web search config | backend=%s | max_results=%s | fetch_top_n=%s | use_cache=%s",
        search_backend,
        max_results,
        fetch_top_n,
        use_cache,
    )

    # ------------------------------------------------------
    # Cases voorbereiden
    # ------------------------------------------------------
    cases: List[Dict[str, Any]] = []
    total_cases = len(case_paths)

    logger.info("Start voorbereiden van %s cases met internetcontext", total_cases)
    t_prepare = time.time()

    for idx, path in enumerate(case_paths, start=1):
        case_file = os.path.basename(path)
        case_id = os.path.splitext(case_file)[0]

        logger.info("Prepare %s/%s | case=%s", idx, total_cases, case_id)

        data = read_case_file(path)
        original_context = data.get("context", "")
        original_metadata = data.get("metadata", {}) or {}

        t_search = time.time()

        if search_backend == "openai":
            # ── OpenAI Responses API met web_search_preview ──
            logger.info("OpenAI web search voor %s...", case_id)
            web_payload = retrieve_web_context_openai(
                case_context=original_context,
                cache_dir=web_cache_dir,
                model=search_model,
                max_results=max_results,
                fetch_top_n=fetch_top_n,
                use_cache=use_cache,
            )
        else:
            # ── Legacy DuckDuckGo (werkt NIET vanuit serverless) ──
            query = build_search_query(
                case_context=original_context,
                query_suffix=query_suffix,
            )
            logger.info("DuckDuckGo query voor %s: %s", case_id, query)
            web_payload = retrieve_web_context(
                query=query,
                cache_dir=web_cache_dir,
                max_results=max_results,
                fetch_top_n=fetch_top_n,
                use_cache=use_cache,
                request_sleep_seconds=request_sleep_seconds,
            )

        n_results = len(web_payload.get("results", []))
        logger.info(
            "Web search klaar | case=%s | backend=%s | resultaten=%s | duur=%.2f sec",
            case_id,
            search_backend,
            n_results,
            time.time() - t_search,
        )

        for rank, item in enumerate(web_payload.get("results", []), start=1):
            logger.info(
                "  web %s | title=%s | url=%s",
                rank,
                item.get("title", "")[:80],
                item.get("url", ""),
            )

        if web_payload.get("web_summary"):
            logger.info(
                "  web_summary: %s chars",
                len(web_payload["web_summary"]),
            )

        augmented_context = build_augmented_web_context(
            case_context=original_context,
            web_payload=web_payload,
        )

        logger.info(
            "Context opgebouwd | case=%s | chars=%s",
            case_id,
            len(augmented_context),
        )

        # Metadata bevat de zoekresultaten + samenvatting
        web_meta = {
            "backend": search_backend,
            "model": search_model if search_backend == "openai" else None,
            "max_results": max_results,
            "fetch_top_n": fetch_top_n,
            "results": web_payload.get("results", []),
            "web_summary_chars": len(web_payload.get("web_summary", "")),
            "original_case_context": original_context,
        }
        if web_payload.get("usage"):
            web_meta["search_usage"] = web_payload["usage"]

        cases.append(
            {
                "case_id": case_id,
                "context": augmented_context,
                "metadata": {
                    **original_metadata,
                    "web_search": web_meta,
                },
            }
        )

        # Rate limiting tussen cases
        if request_sleep_seconds > 0 and idx < total_cases:
            time.sleep(request_sleep_seconds)

    logger.info("Voorbereiden cases klaar | duur=%.2f sec", time.time() - t_prepare)

    # ------------------------------------------------------
    # Runner (zelfde voortgang als experiment A/B)
    # ------------------------------------------------------
    logger.info("Start runs voor experiment_%s", exp_name)

    run_experiment_aggregated_runs(
        experiment_name=exp_name,
        model=model,
        runs=runs,
        chat_params=chat_params,
        cases=cases,
        results_base_dir=results_dir,
        embedding_model=embedding_model,
        logger=logger,
    )

    logger.info("Klaar. Output: %s", exp_output_dir)

    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    return exp_output_dir


if __name__ == "__main__":
    run(project_root=os.getenv("PROJECT_ROOT", "/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu"))
