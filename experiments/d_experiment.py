"""Experiment D: Self-Refine (Iteratieve Zelfreflectie)

Gebaseerd op Madaan et al. (2023) — "Self-Refine: Iterative Refinement
with Self-Feedback"

Architectuur:
  Fase 1 — GENERATE  → Genereer initieel strategisch advies (identiek aan Experiment A)
  Fase 2 — REFLECT   → Gestructureerde kritiek via evaluatierubric
  Fase 3 — REFINE    → Verbeterd advies op basis van de kritiek

Hypothese:
  Iteratieve zelfreflectie levert hogere kwaliteitsscores op dan
  single-pass generatie, maar mogelijk ten koste van consistentie
  (meer variatie door het refinement-proces) en efficiëntie (3× tokens).

Opslagstructuur:
  data/results_data/
    experiment_d/
      run_1.json   # bevat alle cases met alle fasen
      run_2.json
      ...
      summary.json

Gebruik:
  from experiments.d_experiment import run
  run(project_root="/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu")
"""

from __future__ import annotations

import os
import sys
import glob
import json
import time
import shutil
import logging
import importlib
from typing import Dict, Any, Optional, List

# Stap-extractie voor CoT-analyse
try:
    from utils.steps import extract_steps
except ImportError:
    extract_steps = None  # fallback als utils niet beschikbaar


# =========================
# EXPERIMENT SETTINGS (D)
# =========================
EXPERIMENT: Dict[str, Any] = {
    "name": "d",
    "model": "gpt-4o-mini",
    "runs": 5,
    "chat_params": {
        "temperature": 0.8,
        "top_p": 1.0,
        "max_tokens": 10000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "critique_params": {
        "temperature": 0.0,     # deterministische kritiek voor consistentie
        "top_p": 1.0,
        "max_tokens": 2000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
    "refine_params": {
        "temperature": 0.8,     # zelfde creativiteit als initiële generatie
        "top_p": 1.0,
        "max_tokens": 10000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    },
}


# ==========================================================
# SELF-REFINE PROMPTS
# ==========================================================

# ── Fase 2: Critique ──────────────────────────────────────
CRITIQUE_SYSTEM = (
    "Je bent een kritische reviewer van strategische adviezen voor restaurants.\n"
    "Je beoordeelt adviezen op academisch niveau en geeft concrete, "
    "actionable feedback.\n"
    "Je bent streng maar constructief: je benoemt zowel sterke punten als "
    "concrete verbetermogelijkheden.\n"
)

CRITIQUE_TEMPLATE = (
    "Beoordeel het volgende strategisch advies op deze vijf criteria "
    "en geef CONCRETE verbeterpunten:\n\n"

    "1. **Probleemdefinitie**\n"
    "   - Is er één scherp afgebakend strategisch probleem?\n"
    "   - Is het gekwantificeerd met specifieke data uit de context?\n"
    "   - Is de oorzaak-gevolg relatie uitgelegd?\n\n"

    "2. **Analyse**\n"
    "   - Bevat het zowel externe als interne analyse met eigen conclusies?\n"
    "   - Worden concrete datapunten gebruikt (niet alleen geparafraseerd)?\n"
    "   - Worden verbanden gelegd tussen observaties?\n\n"

    "3. **Scenario's**\n"
    "   - Zijn er minstens 2 onderscheidende, plausibele scenario's?\n"
    "   - Zijn de implicaties per scenario concreet uitgewerkt?\n"
    "   - Dragen de scenario's bij aan de strategische keuze?\n\n"

    "4. **Advies**\n"
    "   - Zijn de aanbevelingen concreet en uitvoerbaar?\n"
    "   - Is de gekozen strategie expliciet onderbouwd?\n"
    "   - Zijn risico's en mitigatie benoemd?\n\n"

    "5. **Samenhang (Chain-of-Thought)**\n"
    "   - Bouwt elke stap logisch voort op de vorige?\n"
    "   - Worden inzichten uit de analyse doorvertaald naar het advies?\n"
    "   - Is het geheel coherent van probleem tot implementatie?\n\n"

    "ADVIES OM TE BEOORDELEN:\n"
    "{initial_advice}\n\n"

    "INSTRUCTIE:\n"
    "- Benoem per criterium de 1-2 grootste zwaktes\n"
    "- Geef per zwakte een concrete, specifieke suggestie voor verbetering\n"
    "- Verwijs naar ontbrekende data, vage formuleringen, of logische hiaten\n"
    "- Benoem ook kort wat al goed is (max 1 zin per criterium)\n"
    "- Focus op de meest impactvolle verbeterpunten\n"
)

# ── Fase 3: Refine ────────────────────────────────────────
REFINE_SYSTEM = (
    "Je bent een behulpzame assistent die strategisch advies geeft aan een restaurant.\n"
    "Je werkt op academisch niveau en redeneert gestructureerd en expliciet.\n\n"

    "Je volgt principes uit evidence-based management (Rousseau, 2006):\n"
    "- Adviezen zijn gebaseerd op analyse\n"
    "- Redeneringen zijn expliciet en navolgbaar\n\n"

    "Je werkt volgens een ontwerpgerichte aanpak (Van Aken et al., 2012):\n"
    "- Aannames expliciet maken\n"
    "- Transparant redeneren\n\n"

    "Je hebt eerder een advies geschreven en feedback ontvangen.\n"
    "Je taak is nu om het advies te VERBETEREN op basis van die feedback.\n"
)

REFINE_TEMPLATE = (
    "CONTEXT:\n"
    "{context}\n\n"

    "JE EERDERE ADVIES:\n"
    "{initial_advice}\n\n"

    "KRITIEK VAN DE REVIEWER:\n"
    "{critique}\n\n"

    "INSTRUCTIE:\n"
    "Schrijf een VERBETERDE versie van het strategisch advies die:\n"
    "- Alle concrete verbeterpunten uit de kritiek adresseert\n"
    "- De sterke punten uit het origineel behoudt\n"
    "- Dezelfde 11-stappen structuur volgt\n"
    "- Specifieker en beter onderbouwd is met data uit de context\n"
    "- Sterkere logische verbanden legt tussen analyse en advies\n\n"

    "Gebruik dezelfde structuur maar verbeter de inhoud substantieel.\n"
    "Maak redeneringen expliciet (omdat → daarom → dus).\n"
    "Focus op strategische relevantie en vermijd herhaling.\n"
)


# ==========================================================
# Project root helpers (consistent met a/b/c_experiment.py)
# ==========================================================
def _is_project_root(path: str) -> bool:
    return (
        os.path.isdir(os.path.join(path, "utils"))
        and os.path.isdir(os.path.join(path, "configs"))
        and os.path.isdir(os.path.join(path, "experiments"))
        and os.path.isdir(os.path.join(path, "data"))
    )


def _sanitize_syspath(project_root: str) -> None:
    """Zet project_root vooraan in sys.path en verwijder duplicaten."""
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
    logger = logging.getLogger("experiment_d")
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


# ==========================================================
# Usage helpers
# ==========================================================
def _usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """Converteer OpenAI usage object naar dict."""
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


def _merge_usage(*usages: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregeer token usage over meerdere API calls."""
    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        if u is None:
            continue
        for key in merged:
            merged[key] += int(u.get(key, 0))
    return merged


# ==========================================================
# Self-Refine fasen
# ==========================================================
def phase_generate(*, client, model, context, params, prompts_module):
    """
    Fase 1: Genereer initieel strategisch advies.
    Identiek aan Experiment A — zelfde prompt uit configs/prompts.py.
    """
    messages = prompts_module.build_messages(context=context)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **params,
    )
    latency = time.time() - t0

    return {
        "answer": resp.choices[0].message.content,
        "latency_seconds": latency,
        "usage": _usage_to_dict(getattr(resp, "usage", None)),
        "messages": messages,
    }


def phase_critique(*, client, model, initial_advice, params):
    """
    Fase 2: Gestructureerde kritiek op het advies.
    Gebruikt dezelfde rubric-criteria als de LLM-as-judge evaluatie,
    maar formuleert ze als verbeterpunten in plaats van scores.
    """
    messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM},
        {
            "role": "user",
            "content": CRITIQUE_TEMPLATE.format(
                initial_advice=initial_advice,
            ),
        },
    ]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **params,
    )
    latency = time.time() - t0

    return {
        "critique": resp.choices[0].message.content,
        "latency_seconds": latency,
        "usage": _usage_to_dict(getattr(resp, "usage", None)),
        "messages": messages,
    }


def phase_refine(*, client, model, context, initial_advice, critique, params):
    """
    Fase 3: Verfijnd advies op basis van kritiek.
    Ontvangt het originele advies + de gestructureerde kritiek en
    produceert een verbeterde versie die de zwaktes adresseert.
    """
    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {
            "role": "user",
            "content": REFINE_TEMPLATE.format(
                context=context,
                initial_advice=initial_advice,
                critique=critique,
            ),
        },
    ]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **params,
    )
    latency = time.time() - t0

    return {
        "answer": resp.choices[0].message.content,
        "latency_seconds": latency,
        "usage": _usage_to_dict(getattr(resp, "usage", None)),
        "messages": messages,
    }


# ==========================================================
# Main experiment runner
# ==========================================================
def run(
    project_root: str,
    cases_glob: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    clean_output: bool = True,
) -> str:
    """
    Draait experiment D (Self-Refine).

    Per case worden drie API-calls gedaan:
      1. Generate  — initieel advies (zelfde prompt als Experiment A)
      2. Critique  — gestructureerde kritiek (temperature=0)
      3. Refine    — verbeterd advies op basis van kritiek

    Output: <PROJECT_ROOT>/data/results_data/experiment_d/run_*.json

    Het output-formaat is compatibel met de bestaande analyse-notebook:
    - answer       = het VERFIJNDE advies (fase 3)
    - latency      = totaal over alle drie fasen
    - usage        = geaggregeerde tokens over alle fasen
    - messages     = initiële generatie-messages (voor CGR-compatibiliteit)
    - phases       = nieuw veld met tussenresultaten per fase
    """
    if not project_root:
        raise ValueError("project_root is verplicht.")

    project_root = os.path.abspath(project_root)
    if not _is_project_root(project_root):
        raise ValueError(f"project_root lijkt geen geldige projectmap: {project_root}")

    _sanitize_syspath(project_root)

    # Imports na sys.path sanitize
    from utils.io import read_case_file, write_json
    from utils.openai_client import get_client
    from utils.metrics import mean_pairwise_cosine
    import configs.prompts as prompts
    prompts = importlib.reload(prompts)

    # Paden
    case_dir = os.path.join(project_root, "data", "case_data")
    results_dir = os.path.join(project_root, "data", "results_data")
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Experiment config
    exp_name = EXPERIMENT["name"]
    model = EXPERIMENT["model"]
    runs = int(EXPERIMENT["runs"])
    chat_params = dict(EXPERIMENT["chat_params"])
    critique_params = dict(EXPERIMENT["critique_params"])
    refine_params = dict(EXPERIMENT["refine_params"])

    # Clean output folder
    exp_output_dir = os.path.join(results_dir, f"experiment_{exp_name}")
    if clean_output and os.path.exists(exp_output_dir):
        shutil.rmtree(exp_output_dir)
    os.makedirs(exp_output_dir, exist_ok=True)

    # Logger
    log_path = os.path.join(logs_dir, f"experiment_{exp_name}.log")
    logger = _setup_logger(log_path)

    # Cases laden
    if cases_glob is None:
        cases_glob = os.path.join(case_dir, "case_*.*")
    case_paths = sorted(glob.glob(cases_glob))
    if not case_paths:
        raise ValueError(f"Geen cases gevonden met glob: {cases_glob}")

    cases: List[Dict[str, Any]] = []
    for path in case_paths:
        case_file = os.path.basename(path)
        case_id = os.path.splitext(case_file)[0]
        data = read_case_file(path)
        cases.append({
            "case_id": case_id,
            "context": data.get("context", ""),
            "metadata": data.get("metadata", {}),
        })

    # Logging
    logger.info("==== START experiment_%s (Self-Refine) ====", exp_name)
    logger.info("Project root: %s", project_root)
    logger.info("Using prompts from: %s", prompts.__file__)
    logger.info("PROMPT_ID: %s", prompts.PROMPT_ID)
    logger.info("QUESTION: %s", getattr(prompts, "QUESTION", "<missing>"))
    logger.info("Model: %s", model)
    logger.info("Cases: %s | Runs: %s", len(cases), runs)
    logger.info("Phases: Generate (T=%.1f) → Critique (T=%.1f) → Refine (T=%.1f)",
                chat_params["temperature"],
                critique_params["temperature"],
                refine_params["temperature"])
    logger.info("Output dir: %s", exp_output_dir)

    client = get_client()

    # Track answers per case voor stabiliteitsberekening
    answers_by_case: Dict[str, List[str]] = {c["case_id"]: [] for c in cases}

    # ── Runs uitvoeren ──
    for r in range(1, runs + 1):
        logger.info("── Run %s/%s ──", r, runs)
        run_cases = []

        for case in cases:
            case_id = case["case_id"]
            context = case["context"]

            # ── Fase 1: Generate ──
            logger.info("  %s — Fase 1/3: Generate...", case_id)
            gen = phase_generate(
                client=client,
                model=model,
                context=context,
                params=chat_params,
                prompts_module=prompts,
            )
            gen_tokens = gen["usage"].get("total_tokens", "?") if gen["usage"] else "?"
            logger.info("    → %.1fs | %s tokens", gen["latency_seconds"], gen_tokens)

            # ── Fase 2: Critique ──
            logger.info("  %s — Fase 2/3: Critique...", case_id)
            crit = phase_critique(
                client=client,
                model=model,
                initial_advice=gen["answer"],
                params=critique_params,
            )
            crit_tokens = crit["usage"].get("total_tokens", "?") if crit["usage"] else "?"
            logger.info("    → %.1fs | %s tokens", crit["latency_seconds"], crit_tokens)

            # ── Fase 3: Refine ──
            logger.info("  %s — Fase 3/3: Refine...", case_id)
            ref = phase_refine(
                client=client,
                model=model,
                context=context,
                initial_advice=gen["answer"],
                critique=crit["critique"],
                params=refine_params,
            )
            ref_tokens = ref["usage"].get("total_tokens", "?") if ref["usage"] else "?"
            logger.info("    → %.1fs | %s tokens", ref["latency_seconds"], ref_tokens)

            # ── Aggregeer ──
            total_latency = (
                gen["latency_seconds"]
                + crit["latency_seconds"]
                + ref["latency_seconds"]
            )
            total_usage = _merge_usage(
                gen["usage"],
                crit["usage"],
                ref["usage"],
            )

            final_answer = ref["answer"]
            answers_by_case[case_id].append(final_answer)

            logger.info("  %s — Totaal: %.1fs | %s tokens",
                        case_id, total_latency, total_usage["total_tokens"])

            # Extraheer CoT-stappen uit zowel initieel als verfijnd advies
            _extract = extract_steps if extract_steps else lambda x: {}
            gen_steps = _extract(gen["answer"])
            ref_steps = _extract(ref["answer"])

            # Output in compatibel formaat
            run_cases.append({
                "case_id": case_id,
                "answer": final_answer,              # verfijnd advies
                "steps": ref_steps,                  # ✅ CoT-stappen (verfijnd)
                "latency_seconds": total_latency,    # totaal alle fasen
                "usage": total_usage,                # geaggregeerd
                "messages": gen["messages"],          # initiële messages (CGR-compat)
                "phases": {
                    "generate": {
                        "answer": gen["answer"],
                        "steps": gen_steps,          # ✅ CoT-stappen (initieel)
                        "latency_seconds": gen["latency_seconds"],
                        "usage": gen["usage"],
                    },
                    "critique": {
                        "critique": crit["critique"],
                        "latency_seconds": crit["latency_seconds"],
                        "usage": crit["usage"],
                    },
                    "refine": {
                        "answer": ref["answer"],
                        "steps": ref_steps,          # ✅ CoT-stappen (verfijnd)
                        "latency_seconds": ref["latency_seconds"],
                        "usage": ref["usage"],
                    },
                },
                "metadata": case.get("metadata", {}),
            })

        # Sla run op
        run_payload = {
            "run": r,
            "experiment": exp_name,
            "model": model,
            "technique": "self-refine",
            "prompt_id": prompts.PROMPT_ID,
            "question": prompts.QUESTION,
            "prompts_file": prompts.__file__,
            "chat_params": chat_params,
            "critique_params": critique_params,
            "refine_params": refine_params,
            "cases": run_cases,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        run_path = os.path.join(exp_output_dir, f"run_{r}.json")
        write_json(run_path, run_payload)
        logger.info("  Opgeslagen: %s", run_path)

    # ── Stabiliteitsberekening ──
    logger.info("── Stabiliteitsberekening ──")

    stability_by_case = {}
    for case_id, answers in answers_by_case.items():
        if len(answers) >= 2:
            resp = client.embeddings.create(
                model=embedding_model,
                input=answers,
            )
            embeddings = [item.embedding for item in resp.data]
            s = mean_pairwise_cosine(embeddings)
            stability_by_case[case_id] = s
            logger.info("  S(%s) = %.4f", case_id, s)

    avg_stability = (
        sum(stability_by_case.values()) / len(stability_by_case)
        if stability_by_case else 0.0
    )
    logger.info("  S(gemiddeld) = %.4f", avg_stability)

    # ── Summary ──
    summary = {
        "experiment": exp_name,
        "technique": "self-refine",
        "model": model,
        "prompt_id": prompts.PROMPT_ID,
        "question": prompts.QUESTION,
        "runs": runs,
        "cases": len(cases),
        "phases": ["generate", "critique", "refine"],
        "temperature_per_phase": {
            "generate": chat_params["temperature"],
            "critique": critique_params["temperature"],
            "refine": refine_params["temperature"],
        },
        "embedding_model": embedding_model,
        "stability_by_case": stability_by_case,
        "avg_stability": avg_stability,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(os.path.join(exp_output_dir, "summary.json"), summary)

    logger.info("==== KLAAR experiment_%s ====", exp_name)
    logger.info("Output: %s", exp_output_dir)
    return exp_output_dir


if __name__ == "__main__":
    run(
        project_root=os.getenv(
            "PROJECT_ROOT",
            "/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu",
        )
    )
