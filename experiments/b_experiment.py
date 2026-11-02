"""
Experiment B: Baseline + RAG (menu engineering + scenario theory JSON)

Opslagstructuur:

data/results_data/
  experiment_b/
    run_1.json
    run_2.json
    ...
    summary.json

Vereiste kennisbronnen:
- data/rag_data/menu_engineering/*.txt / *.md
- JSON-bestand met scenario-analyse theorie

Gebruik:
from experiments.b_experiment import run
run(project_root="/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu")
"""

from __future__ import annotations

import os
import sys
import glob
import time
import json
import shutil
import logging
from typing import Dict, Any, Optional, List


# =========================
# EXPERIMENT SETTINGS (B)
# =========================
EXPERIMENT: Dict[str, Any] = {
    "name": "b",
    "model": "gpt-4o-mini",
    "runs": 5,
    "chat_params": {
        "temperature": 0.8,
        "top_p": 1.0,
        "max_tokens": 10000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "rag": {
        # gegarandeerde verdeling per bron
        "menu_top_k": 2,
        "scenario_top_k": 2,
        "extra_top_k": 1,
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1200,
        "overlap": 150,
        "use_cache": True,
        "json_source": "/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu/data/rag_data/scenario_theory/theorie.JSON",
        "generated_theory_dir": "data/rag_data/generated_RAG_scenario_theory",
        "rebuild_generated_theory": True,
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
    logger_name = f"experiment_b_logger_{os.path.abspath(log_path)}"
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
    return logger


# ==========================================================
# JSON -> TXT bronbestanden voor RAG
# ==========================================================
def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    return json.dumps(value, ensure_ascii=False, indent=2)


def _slugify(value: str) -> str:
    value = (value or "").strip().lower()
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def _theory_entry_to_text(entry: Dict[str, Any]) -> str:
    parts: List[str] = []

    title = entry.get("theory_name") or entry.get("theory_id") or "Onbekende theorie"
    parts.append(f"theory_name:\n{title}")

    fields_in_order = [
        "reference",
        "primary_purpose",
        "when_to_use",
        "retrieval_queries",
        "key_concepts",
        "how_the_llm_should_consult_theory",
        "how_the_llm_should_apply_theory",
        "llm_operational_steps",
        "output_markers",
        "pitfalls_to_avoid",
        "best_used_with",
    ]

    for field in fields_in_order:
        if field not in entry:
            continue
        rendered = _safe_text(entry[field])
        if rendered:
            parts.append(f"{field}:\n{rendered}")

    return "\n\n".join(parts).strip()


def _write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _materialize_json_rag_sources(
    json_path: str,
    output_dir: str,
    rebuild: bool = True,
) -> List[str]:
    """
    Zet een JSON theoriebron om naar losse .txt bestanden zodat build_chunk_index()
    exact hetzelfde chunk-type maakt als voor de gewone file-based RAG-bronnen.
    """
    if rebuild and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    written_files: List[str] = []

    theories = data.get("theories", [])
    for idx, theory in enumerate(theories, start=1):
        theory_name = theory.get("theory_name") or theory.get("theory_id") or f"theory_{idx}"
        theory_slug = _slugify(theory_name)
        file_path = os.path.join(output_dir, f"{idx:02d}_{theory_slug}.txt")
        content = _theory_entry_to_text(theory)
        if content.strip():
            _write_text_file(file_path, content)
            written_files.append(file_path)

    global_instructions = data.get("global_instructions")
    if global_instructions:
        file_path = os.path.join(output_dir, "00_global_instructions.txt")
        content = "global_instructions:\n" + _safe_text(global_instructions)
        _write_text_file(file_path, content)
        written_files.append(file_path)

    rag_flow = data.get("recommended_rag_flow")
    if rag_flow:
        file_path = os.path.join(output_dir, "99_recommended_rag_flow.txt")
        content = "recommended_rag_flow:\n" + _safe_text(rag_flow)
        _write_text_file(file_path, content)
        written_files.append(file_path)

    manifest = {
        "source_json": json_path,
        "generated_files": written_files,
        "generated_at_unix": time.time(),
    }
    manifest_path = os.path.join(output_dir, "_manifest.json")
    _write_text_file(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    return written_files


# ==========================================================
# Retrieval helpers
# ==========================================================
def _chunk_id(chunk: Any) -> str:
    return str(getattr(chunk, "chunk_id", None) or "")


def _chunk_source(chunk: Any) -> str:
    return str(getattr(chunk, "source_file", None) or "")


def _chunk_score(chunk: Any) -> float:
    score = getattr(chunk, "score", 0.0)
    try:
        return float(score or 0.0)
    except Exception:
        return 0.0


def _dedupe_chunks(chunks: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for chunk in chunks:
        key = _chunk_id(chunk)
        if not key:
            key = f"{_chunk_source(chunk)}::{id(chunk)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _sort_chunks_by_score_desc(chunks: List[Any]) -> List[Any]:
    return sorted(chunks, key=_chunk_score, reverse=True)


# ==========================================================
# Main
# ==========================================================
def run(
    project_root: str,
    cases_glob: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    clean_output: bool = True,
    json_rag_path: Optional[str] = None,
) -> str:
    """
    Draait experiment B:
    - haalt per case relevante kennis op uit menu engineering
    - gebruikt scenario-theorie JSON
    - zet JSON om naar tekstbestanden
    - indexeert beide bronnen met dezelfde build_chunk_index()-flow
    - retrieved PER BRON zodat scenario-theorie gegarandeerd meedoet
    - draait daarna dezelfde runner als experiment A

    Output:
    <PROJECT_ROOT>/data/results_data/experiment_b/run_*.json
    """
    if not project_root:
        raise ValueError("project_root is verplicht.")

    project_root = os.path.abspath(project_root)
    if not _is_project_root(project_root):
        raise ValueError(f"project_root lijkt geen geldige projectmap: {project_root}")

    _sanitize_syspath(project_root)

    from utils.io import read_case_file
    from utils.runner import run_experiment_aggregated_runs
    from utils.rag import build_chunk_index, retrieve_top_k, build_augmented_context

    import configs.prompts as prompts
    import importlib
    prompts = importlib.reload(prompts)

    case_dir = os.path.join(project_root, "data", "case_data")
    menu_rag_dir = os.path.join(project_root, "data", "rag_data", "menu_engineering")
    results_dir = os.path.join(project_root, "data", "results_data")
    logs_dir = os.path.join(project_root, "logs")

    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    exp_name = EXPERIMENT["name"]
    model = EXPERIMENT["model"]
    runs = int(EXPERIMENT["runs"])
    chat_params = dict(EXPERIMENT["chat_params"])

    rag_cfg = dict(EXPERIMENT["rag"])
    menu_top_k = int(rag_cfg["menu_top_k"])
    scenario_top_k = int(rag_cfg["scenario_top_k"])
    extra_top_k = int(rag_cfg.get("extra_top_k", 0))
    rag_embedding_model = str(rag_cfg["embedding_model"])
    rag_chunk_size = int(rag_cfg["chunk_size"])
    rag_overlap = int(rag_cfg["overlap"])
    rag_use_cache = bool(rag_cfg.get("use_cache", True))
    generated_theory_dir = str(
        rag_cfg.get("generated_theory_dir", "data/rag_data/_generated_scenario_theory")
    )
    rebuild_generated_theory = bool(rag_cfg.get("rebuild_generated_theory", True))

    configured_json_source = rag_cfg.get("json_source")
    if json_rag_path is None:
        json_rag_path = configured_json_source

    if json_rag_path and not os.path.isabs(json_rag_path):
        json_rag_path = os.path.join(project_root, json_rag_path)

    if not json_rag_path:
        raise ValueError("Scenario-theorie moet gebruikt worden, maar json_source is leeg of None.")

    if not os.path.isabs(generated_theory_dir):
        generated_theory_dir = os.path.join(project_root, generated_theory_dir)

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
    logger.info("Output dir: %s", exp_output_dir)
    logger.info(
        "RAG config | menu_top_k=%s | scenario_top_k=%s | extra_top_k=%s | embedding_model=%s | chunk_size=%s | overlap=%s | use_cache=%s",
        menu_top_k,
        scenario_top_k,
        extra_top_k,
        rag_embedding_model,
        rag_chunk_size,
        rag_overlap,
        rag_use_cache,
    )

    # ------------------------------------------------------
    # Scenario JSON verplicht aanwezig
    # ------------------------------------------------------
    logger.info("JSON scenario-bron ingesteld: %s", json_rag_path)
    logger.info("JSON bestaat: %s", os.path.exists(json_rag_path))
    if not os.path.exists(json_rag_path):
        raise ValueError(
            f"Scenario-theorie moet gebruikt worden, maar JSON RAG-bestand is niet gevonden: {json_rag_path}"
        )

    t_materialize = time.time()
    generated_files = _materialize_json_rag_sources(
        json_path=json_rag_path,
        output_dir=generated_theory_dir,
        rebuild=rebuild_generated_theory,
    )
    logger.info(
        "Scenario-theorie gegenereerd naar tekstbestanden | files=%s | dir=%s | duur=%.2f sec",
        len(generated_files),
        generated_theory_dir,
        time.time() - t_materialize,
    )

    # ------------------------------------------------------
    # Index per bron bouwen
    # ------------------------------------------------------
    if not os.path.isdir(menu_rag_dir):
        raise ValueError(f"Menu-engineering RAG-map niet gevonden: {menu_rag_dir}")

    logger.info("Gebruik file-based RAG-bron: %s", menu_rag_dir)
    t_index = time.time()
    menu_chunks = build_chunk_index(
        rag_folder=menu_rag_dir,
        embedding_model=rag_embedding_model,
        chunk_size=rag_chunk_size,
        overlap=rag_overlap,
        use_cache=rag_use_cache,
    )
    logger.info(
        "Menu-engineering chunk-index klaar | chunks=%s | duur=%.2f sec",
        len(menu_chunks),
        time.time() - t_index,
    )

    if not os.path.isdir(generated_theory_dir):
        raise ValueError(f"Gegenereerde scenario-theorie map niet gevonden: {generated_theory_dir}")

    logger.info("Gebruik gegenereerde scenario-theorie RAG-bron: %s", generated_theory_dir)
    t_index = time.time()
    scenario_chunks = build_chunk_index(
        rag_folder=generated_theory_dir,
        embedding_model=rag_embedding_model,
        chunk_size=rag_chunk_size,
        overlap=rag_overlap,
        use_cache=rag_use_cache,
    )
    logger.info(
        "Scenario-theorie chunk-index klaar | chunks=%s | duur=%.2f sec",
        len(scenario_chunks),
        time.time() - t_index,
    )

    if not menu_chunks:
        raise ValueError("Geen menu-engineering chunks gevonden.")
    if not scenario_chunks:
        raise ValueError("Geen scenario-theorie chunks gevonden.")

    combined_chunk_index: List[Any] = []
    combined_chunk_index.extend(menu_chunks)
    combined_chunk_index.extend(scenario_chunks)

    logger.info(
        "Gecombineerde RAG-index klaar | totaal_chunks=%s | bronnen=%s",
        len(combined_chunk_index),
        "menu_engineering, scenario_theory_generated",
    )

    # ------------------------------------------------------
    # Cases voorbereiden
    # ------------------------------------------------------
    cases: List[Dict[str, Any]] = []
    total_cases = len(case_paths)

    logger.info("Start voorbereiden van %s cases met gecombineerde RAG-context", total_cases)
    t_prepare = time.time()

    for idx, path in enumerate(case_paths, start=1):
        case_file = os.path.basename(path)
        case_id = os.path.splitext(case_file)[0]

        logger.info("Prepare %s/%s | case=%s", idx, total_cases, case_id)

        data = read_case_file(path)
        original_context = data.get("context", "")
        original_metadata = data.get("metadata", {}) or {}

        t_retrieval = time.time()

        retrieved_menu_chunks = retrieve_top_k(
            query=original_context,
            chunks=menu_chunks,
            top_k=menu_top_k,
            embedding_model=rag_embedding_model,
        )

        retrieved_scenario_chunks = retrieve_top_k(
            query=original_context,
            chunks=scenario_chunks,
            top_k=scenario_top_k,
            embedding_model=rag_embedding_model,
        )

        retrieved_extra_chunks: List[Any] = []
        if extra_top_k > 0:
            retrieved_extra_chunks = retrieve_top_k(
                query=original_context,
                chunks=combined_chunk_index,
                top_k=extra_top_k,
                embedding_model=rag_embedding_model,
            )

        retrieved_chunks = _dedupe_chunks(
            retrieved_menu_chunks + retrieved_scenario_chunks + retrieved_extra_chunks
        )
        retrieved_chunks = _sort_chunks_by_score_desc(retrieved_chunks)

        logger.info(
            "RAG klaar | case=%s | total_chunks=%s | menu_chunks=%s | scenario_chunks=%s | extra_chunks=%s | duur=%.2f sec",
            case_id,
            len(retrieved_chunks),
            len(retrieved_menu_chunks),
            len(retrieved_scenario_chunks),
            len(retrieved_extra_chunks),
            time.time() - t_retrieval,
        )

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            chunk_id = getattr(chunk, "chunk_id", None)
            source_file = getattr(chunk, "source_file", None)
            score = getattr(chunk, "score", 0.0)
            text = getattr(chunk, "text", "")

            logger.info(
                "  chunk %s | chunk_id=%s | source=%s | score=%.4f | text_len=%s",
                rank,
                chunk_id,
                source_file,
                float(score or 0.0),
                len(text or ""),
            )

        augmented_context = build_augmented_context(
            case_context=original_context,
            retrieved_chunks=retrieved_chunks,
        )

        logger.info("Context opgebouwd | case=%s | chars=%s", case_id, len(augmented_context))

        cases.append(
            {
                "case_id": case_id,
                "context": augmented_context,
                "metadata": {
                    **original_metadata,
                    "retrieval": {
                        "rag_sources": ["menu_engineering", "scenario_theory_generated"],
                        "menu_top_k": menu_top_k,
                        "scenario_top_k": scenario_top_k,
                        "extra_top_k": extra_top_k,
                        "embedding_model": rag_embedding_model,
                        "json_rag_path": json_rag_path,
                        "generated_theory_dir": generated_theory_dir,
                        "retrieved_chunks_count": len(retrieved_chunks),
                        "original_case_context": original_context,
                    },
                },
            }
        )

    logger.info("Voorbereiden cases klaar | duur=%.2f sec", time.time() - t_prepare)

    # ------------------------------------------------------
    # Runner
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
    return exp_output_dir


if __name__ == "__main__":
    run(project_root=os.getenv("PROJECT_ROOT", "/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu"))