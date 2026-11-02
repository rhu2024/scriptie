"""
Hypertune.py — LLM Parameter Sensitivity Analysis

Systematische vergelijking van LLM-parameterinstellingen per architectuur.
Test of de experiment-ranking robuust is over verschillende temperatures.

Aanpak:
  - Hergebruikt de EXACTE messages (incl. RAG/web context) uit bestaande runs
  - Varieert ALLEEN temperature (top_p vast op 1.0)
  - Zo wordt het effect van temperature geïsoleerd van retrieval-randomness
  - Beoordeling: strenge 0–100 judge (identiek aan result_analyses)

Experimenten:
  A = Baseline (prompt + case context)
  B = RAG (menu engineering + scenario theorie context)
  C = Web search (real-time internet context)
  D = Self-Refine (generate → critique → refine)

Output:
  - data/results_data/hypertune_results.csv
  - data/results_data/hypertune_answers.json  (ruwe antwoorden voor downstream analyse)
  - Sensitivity: verandert de ranking bij andere temperatures?
"""

import itertools
import json
import re
import time
import os
import sys

import numpy as np
import pandas as pd

# =========================
# PROJECT SETUP
# =========================
PROJECT_ROOT = "/Workspace/Users/rgoedings@hanos.nl/AI_op_het_menu"
sys.path.insert(0, PROJECT_ROOT)

from utils.openai_client import get_client

client = get_client()

# =========================
# CONFIG
# =========================
MODEL = "gpt-4o-mini"
N_RUNS = 3           # runs per config (3 genoeg voor sensitivity)
TOP_P = 1.0          # vast

# Welke cases testen (2 representatief: eetcafé + gastronomisch)
TEST_CASES = ["case_1", "case_3"]

# Alleen temperature variëren
TEMPERATURES = [0.3, 0.5, 0.8, 1.0]

EXPERIMENTS = ["a", "b", "c", "d"]
EXP_LABELS = {"a": "Baseline", "b": "RAG", "c": "Web search", "d": "Self-Refine"}

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results_data")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "hypertune_results.csv")
OUTPUT_ANSWERS = os.path.join(RESULTS_DIR, "hypertune_answers.json")


# =========================
# MESSAGES LADEN UIT BESTAANDE RUNS
# =========================
def load_messages_from_runs() -> dict:
    """
    Laad de exacte messages (incl. RAG/web context) uit de bestaande run_1.json.
    Returns: {(experiment, case_id): messages_list}
    """
    msg_cache = {}
    for exp in EXPERIMENTS:
        run_path = os.path.join(RESULTS_DIR, f"experiment_{exp}", "run_1.json")
        if not os.path.exists(run_path):
            print(f"  \u26a0\ufe0f  {run_path} niet gevonden, skip {exp}")
            continue
        with open(run_path) as f:
            data = json.load(f)
        for case in data.get("cases", []):
            case_id = case["case_id"]
            messages = case.get("messages", [])
            if messages:
                msg_cache[(exp, case_id)] = messages
    return msg_cache


# =========================
# JSON PARSING
# =========================
def parse_judge_json(text: str) -> dict:
    if not text or not text.strip():
        return {}
    text = text.strip()
    if "```" in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r',\s*([}\]])', r'\1', text))
        except json.JSONDecodeError:
            return {}


# =========================
# JUDGE (streng, 0–100)
# =========================
SCORE_KEYS = ["probleemdefinitie", "analyse", "scenarios", "advies", "cot_samenhang"]

JUDGE_PROMPT = """Je bent een ZEER STRENGE academische beoordelaar van strategische adviezen.
Beoordeel op vijf criteria (0–100 per criterium).

CRITERIATABEL:
|  0–15: Onvoldoende | 16–30: Zwak | 31–45: Matig (generiek) |
| 46–58: Voldoende   | 59–70: Ruim voldoende (concreet) |
| 71–80: Goed        | 81–90: Zeer goed | 91–100: Uitmuntend (bijna onmogelijk) |

REGELS:
- Generiek advies (toepasbaar op elk restaurant) = MAXIMAAL 45.
- Specifieke data nodig voor >58. Scores boven 80 vereisen KPI's + tijdlijnen.
- Gebruik GEEN ronde getallen. Gemiddeld advies hoort rond 45–60.

CRITERIA:
1. PROBLEEMDEFINITIE (0–100): Scherp, afgebakend, gekwantificeerd?
2. ANALYSE (0–100): Diep, verbindend, concrete data, eigen conclusies?
3. SCENARIOS (0–100): Onderscheidend, plausibel, passend bij restaurantcontext?
4. ADVIES (0–100): Concreet, uitvoerbaar, prioritering, KPI's?
5. COT_SAMENHANG (0–100): Logische doorvertaling van begin tot eind?

ADVIES:
{text}

Geef ALLEEN valid JSON:
{{
  "probleemdefinitie": {{"score": <0-100>}},
  "analyse":           {{"score": <0-100>}},
  "scenarios":         {{"score": <0-100>}},
  "advies":            {{"score": <0-100>}},
  "cot_samenhang":     {{"score": <0-100>}}
}}
"""

def judge_output(text: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(text=text[:5000])}],
            temperature=0, max_tokens=500,
        )
        parsed = parse_judge_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"    Judge error: {e}")
        parsed = {}

    scores = {}
    for key in SCORE_KEYS:
        crit_data = parsed.get(key, {})
        if isinstance(crit_data, dict):
            score = crit_data.get("score", 40)
        elif isinstance(crit_data, (int, float)):
            score = crit_data
        else:
            score = 40
        scores[key] = max(0, min(100, int(score)))
    return scores


# =========================
# CONSISTENCY
# =========================
def get_embeddings(texts: list) -> list:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(item.embedding) for item in resp.data]

def mean_pairwise_cosine(embeddings: list) -> float:
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a, b = embeddings[i], embeddings[j]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
    return np.mean(sims) if sims else 0.0


# =========================
# GENERATIE FUNCTIES
# =========================
def generate_standard(messages: list, params: dict) -> tuple:
    """Experimenten A, B, C: directe generatie met opgeslagen messages."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, **params
    )
    latency = time.time() - t0
    answer = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return answer, latency, tokens

def generate_self_refine(messages: list, params: dict) -> tuple:
    """Experiment D: generate \u2192 critique \u2192 refine."""
    t0 = time.time()
    total_tokens = 0

    # Phase 1: Generate
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, **params
    )
    initial = resp.choices[0].message.content or ""
    total_tokens += resp.usage.total_tokens if resp.usage else 0

    # Phase 2: Critique (altijd temp=0 voor consistente feedback)
    critique_prompt = (
        "Analyseer het volgende strategisch advies KRITISCH. "
        "Benoem zwakke punten, ontbrekende analyses, en logische hiaten.\n\n"
        f"ADVIES:\n{initial[:3000]}\n\n"
        "Geef een gestructureerde kritiek met concrete verbeterpunten."
    )
    resp2 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0, top_p=1.0, max_tokens=2000,
    )
    critique = resp2.choices[0].message.content or ""
    total_tokens += resp2.usage.total_tokens if resp2.usage else 0

    # Phase 3: Refine (zelfde temp als generate)
    refine_prompt = (
        "Verbeter het onderstaande advies op basis van de kritiek. "
        "Behoud wat goed is, verbeter wat zwak is.\n\n"
        f"ORIGINEEL ADVIES:\n{initial[:2500]}\n\n"
        f"KRITIEK:\n{critique[:1500]}\n\n"
        "Geef het VERBETERDE advies:"
    )
    resp3 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": refine_prompt}],
        **params
    )
    answer = resp3.choices[0].message.content or ""
    total_tokens += resp3.usage.total_tokens if resp3.usage else 0
    latency = time.time() - t0

    return answer, latency, total_tokens


# =========================
# MAIN EXPERIMENT LOOP
# =========================
def run_tuning():
    """Draait de sensitivity analyse: hergebruikt bestaande contexts, varieert temperature."""

    print("=" * 70)
    print("LLM TEMPERATURE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Experimenten: {EXPERIMENTS}")
    print(f"Cases: {TEST_CASES}")
    print(f"Runs per config: {N_RUNS}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"top_p: {TOP_P} (vast)")

    # Stap 1: Laad messages uit bestaande runs
    print("\nMessages laden uit bestaande run files...")
    msg_cache = load_messages_from_runs()
    print(f"  Geladen: {len(msg_cache)} (experiment, case) combinaties")
    for (exp, case_id), msgs in sorted(msg_cache.items()):
        total_chars = sum(len(m["content"]) for m in msgs)
        print(f"    {exp}/{case_id}: {len(msgs)} messages, {total_chars} chars")

    # Bereken totalen
    n_combos = sum(1 for e in EXPERIMENTS for c in TEST_CASES if (e, c) in msg_cache)
    total_gen = len(TEMPERATURES) * n_combos * N_RUNS
    d_cases = sum(1 for c in TEST_CASES if ("d", c) in msg_cache)
    extra_d = len(TEMPERATURES) * d_cases * N_RUNS * 2  # critique + refine
    total_api = total_gen + total_gen + extra_d  # gen + judge + D extras

    print(f"\nTotaal generaties: {total_gen}")
    print(f"Geschatte API calls: ~{total_api}")
    print(f"Geschatte tijd: ~{total_api * 3 / 60:.0f} minuten")
    print("=" * 70)

    results = []
    all_answers = {}  # Ruwe antwoorden opslaan voor downstream analyse (scenario diversiteit etc.)
    trial_id = 0

    for temp_idx, temp in enumerate(TEMPERATURES, 1):
        params = {"temperature": temp, "top_p": TOP_P, "max_tokens": 10000}

        for exp in EXPERIMENTS:
            generator = generate_self_refine if exp == "d" else generate_standard

            for case_id in TEST_CASES:
                if (exp, case_id) not in msg_cache:
                    print(f"  \u26a0\ufe0f  Skip {exp}/{case_id}: geen messages")
                    continue

                trial_id += 1
                messages = msg_cache[(exp, case_id)]

                print(f"\n[{trial_id}] temp {temp_idx}/{len(TEMPERATURES)} | "
                      f"exp={exp} ({EXP_LABELS[exp]}) | case={case_id} | "
                      f"temp={temp}")

                outputs = []
                judge_scores_list = []
                latencies = []
                token_counts = []

                for run in range(N_RUNS):
                    try:
                        answer, latency, tokens = generator(messages, params)
                        outputs.append(answer)
                        latencies.append(latency)
                        token_counts.append(tokens)

                        scores = judge_output(answer)
                        judge_scores_list.append(scores)

                        total = round(np.mean(list(scores.values())), 1)
                        print(f"  run {run+1}: {total}/100 | {latency:.1f}s | {tokens} tokens")
                    except Exception as e:
                        print(f"  run {run+1} FAILED: {e}")
                        continue

                if not outputs:
                    continue

                # Sla ruwe antwoorden op per trial
                answer_key = f"{exp}_{case_id}_temp{temp}"
                all_answers[answer_key] = {
                    "experiment": exp,
                    "case_id": case_id,
                    "temperature": temp,
                    "answers": outputs,
                }

                # Metrics
                avg_scores = {k: np.mean([s[k] for s in judge_scores_list]) for k in SCORE_KEYS}
                judge_total = np.mean(list(avg_scores.values()))

                embeddings = get_embeddings(outputs)
                stability = mean_pairwise_cosine(embeddings)

                avg_word_count = np.mean([len(o.split()) for o in outputs])

                results.append({
                    "trial_id": trial_id,
                    "experiment": exp,
                    "experiment_name": EXP_LABELS[exp],
                    "case_id": case_id,
                    "temperature": temp,
                    "n_runs": len(outputs),
                    **{f"judge_{k}": round(v, 1) for k, v in avg_scores.items()},
                    "judge_total": round(judge_total, 1),
                    "stability": round(stability, 4),
                    "avg_latency": round(np.mean(latencies), 1),
                    "avg_tokens": round(np.mean(token_counts), 0),
                    "avg_words": round(avg_word_count, 0),
                })

    # =========================
    # OPSLAAN
    # =========================
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nResultaten opgeslagen: {OUTPUT_CSV}")
    print(f"Totaal trials: {len(df)}")

    # Sla ruwe antwoorden op als JSON (voor scenario-diversiteit analyse)
    with open(OUTPUT_ANSWERS, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    print(f"Ruwe antwoorden opgeslagen: {OUTPUT_ANSWERS} ({len(all_answers)} trials)")

    # =========================
    # SAMENVATTING
    # =========================
    print("\n" + "=" * 70)
    print("SENSITIVITY SAMENVATTING")
    print("=" * 70)

    # Gemiddelde score per experiment \u00d7 temperature
    pivot = df.pivot_table(
        index="experiment", columns="temperature",
        values="judge_total", aggfunc="mean"
    ).round(1)
    print("\nGemiddelde judge score per experiment \u00d7 temperature:")
    print(pivot.to_string())

    # Ranking per temperature
    print("\nRanking per temperature (van best naar slechtst):")
    for temp in TEMPERATURES:
        sub = df[df["temperature"] == temp].groupby("experiment")["judge_total"].mean()
        ranking = sub.sort_values(ascending=False)
        rank_str = " > ".join(f"{e.upper()}({v:.1f})" for e, v in ranking.items())
        print(f"  temp={temp}: {rank_str}")

    # Stabiliteit per experiment \u00d7 temperature
    print("\nStabiliteit per experiment \u00d7 temperature:")
    stab_pivot = df.pivot_table(
        index="experiment", columns="temperature",
        values="stability", aggfunc="mean"
    ).round(3)
    print(stab_pivot.to_string())

    # Robuustheid
    print("\nRobuustheid analyse:")
    rankings_per_temp = []
    for temp in TEMPERATURES:
        sub = df[df["temperature"] == temp]
        if sub.empty:
            continue
        rank = sub.groupby("experiment")["judge_total"].mean().rank(ascending=False)
        rankings_per_temp.append(rank)

    if rankings_per_temp:
        rank_df = pd.DataFrame(rankings_per_temp)
        for exp in EXPERIMENTS:
            if exp in rank_df.columns:
                ranks = rank_df[exp]
                print(f"  {exp.upper()} ({EXP_LABELS[exp]}): "
                      f"gem. rank = {ranks.mean():.2f}, "
                      f"std = {ranks.std():.2f}, "
                      f"range = [{ranks.min():.0f}\u2013{ranks.max():.0f}]")

    return df


if __name__ == "__main__":
    run_tuning()
