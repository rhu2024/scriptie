"""
Experiment A: Baseline
Opslagstructuur:

data/results_data/
  experiment_a/
    run_1.json   # bevat alle cases
    run_2.json
    ...
    summary.json

Gebruik:
from experiments.a_experiment import run
run(project_root="/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu")
"""

from __future__ import annotations

import os
import sys
import glob
import logging
from typing import Dict, Any, Optional, List


# =========================
# EXPERIMENT SETTINGS (A)
# =========================
EXPERIMENT: Dict[str, Any] = {
    "name": "a",
    "model": "gpt-4o-mini",
    "runs": 5,
    "chat_params": {
        "temperature": 0.8,
        "top_p": 1.0,
        "max_tokens": 10000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
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
    """
    Cruciaal in Databricks:
    - verwijder andere paden die (een andere kopie van) AI_op_het_menu bevatten
    - zet jouw project_root vooraan zodat imports altijd daar vandaan komen
    """
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
    logger = logging.getLogger("experiment_a")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run(
    project_root: str,
    cases_glob: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    clean_output: bool = True,
) -> str:
    """
    Draait experiment A.
    Resultaten: <PROJECT_ROOT>/data/results_data/experiment_a/run_*.json

    - Cases bevatten alleen context/user info
    - Vaste onderzoeksvraag staat in configs/prompts.py (QUESTION)
    """
    if not project_root:
        raise ValueError("project_root is verplicht.")

    project_root = os.path.abspath(project_root)
    if not _is_project_root(project_root):
        raise ValueError(f"project_root lijkt geen geldige projectmap: {project_root}")

    _sanitize_syspath(project_root)

    # Imports pas na sys.path sanitize
    from utils.io import read_case_file
    from utils.runner import run_experiment_aggregated_runs
    import configs.prompts as prompts
    import importlib
    prompts = importlib.reload(prompts)

    case_dir = os.path.join(project_root, "data", "case_data")
    results_dir = os.path.join(project_root, "data", "results_data")
    logs_dir = os.path.join(project_root, "logs")

    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    exp_name = EXPERIMENT["name"]
    model = EXPERIMENT["model"]
    runs = int(EXPERIMENT["runs"])
    chat_params = dict(EXPERIMENT["chat_params"])

    # 🔥 CLEAN OUTPUT FOLDER
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

    # cases inlezen -> lijst dicts (ALLEEN context)
    cases: List[Dict[str, Any]] = []
    for path in case_paths:
        case_file = os.path.basename(path)
        case_id = os.path.splitext(case_file)[0]
        data = read_case_file(path)

        cases.append(
            {
                "case_id": case_id,
                "context": data.get("context", ""),
                "metadata": data.get("metadata", {}),
            }
        )

    logger.info("==== START experiment_%s ====", exp_name)
    logger.info("Project root: %s", project_root)
    logger.info("Using prompts from: %s", prompts.__file__)
    logger.info("PROMPT_ID: %s", prompts.PROMPT_ID)
    logger.info("QUESTION: %s", getattr(prompts, "QUESTION", "<missing QUESTION in prompts.py>"))
    logger.info("Cases: %s", len(cases))
    logger.info("Output dir: %s", exp_output_dir)

    # Let op: jouw runner moet nu géén question verwachten in cases
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
    return exp_output_dir


if __name__ == "__main__":
    run(project_root=os.getenv("PROJECT_ROOT", "/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu"))