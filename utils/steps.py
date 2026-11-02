"""Shared step extraction for Chain-of-Thought (CoT) analysis.

De prompt instrueert het model om 11 gestructureerde redeneerstappen te volgen.
Dit module extraheert en labelt elke stap uit het gegenereerde antwoord,
zodat ze individueel opgeslagen en beoordeeld kunnen worden.

Gebruik:
    from utils.steps import extract_steps, COT_STEPS, STEP_LABELS
    steps = extract_steps(answer_text)
    # => {"1": {"label": "Probleemdefinitie en doel", "text": "..."}, ...}
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# De 11 gestructureerde redeneerstappen uit de prompt
COT_STEPS: List[Tuple[str, str]] = [
    ("1", "Probleemdefinitie en doel"),
    ("2", "Externe analyse"),
    ("3", "Interne analyse"),
    ("4", "Synthese"),
    ("5", "Scenario-analyse"),
    ("6", "Strategische opties"),
    ("7", "Evaluatie en keuze"),
    ("8", "Strategisch advies"),
    ("9", "Implementatie"),
    ("10", "Monitoring en bijsturing"),
    ("11", "Aannames en beperkingen"),
]

STEP_LABELS: Dict[str, str] = {num: label for num, label in COT_STEPS}


def extract_steps(answer: str) -> Dict[str, Dict[str, str]]:
    """
    Extraheer individuele redeneerstappen uit een gestructureerd antwoord.

    Het antwoord bevat typisch markdown headers zoals:
        ### 1. Probleemdefinitie en doel
    of:
        **1. Probleemdefinitie en doel**

    Parameters:
        answer: Het volledige antwoord van het model.

    Returns:
        Dict met stapnummer (str) als key en {"label": str, "text": str} als value.
        Voorbeeld: {"1": {"label": "Probleemdefinitie en doel", "text": "..."}}
        Retourneert een lege dict als geen stappen gevonden worden.
    """
    if not answer or not isinstance(answer, str):
        return {}

    # Vind alle sectie-headers met positie en stapnummer
    header_pattern = r'(?:^|\n)\s*((?:#{1,3}|\*\*)\s*(\d{1,2})\.\s.+)'
    headers = list(re.finditer(header_pattern, answer))

    if not headers:
        return {}

    # Bouw geordende lijst: (positie, stapnummer, header_tekst)
    found = []
    for h in headers:
        num = h.group(2)
        found.append((h.start(), num, h.group(1).strip()))
    found.sort(key=lambda x: x[0])

    # Extraheer tekst per stap (van header tot volgende header)
    steps: Dict[str, Dict[str, str]] = {}
    for i, (pos, num, header) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(answer)
        label = STEP_LABELS.get(num, f"Stap {num}")
        text = answer[pos:end].strip()
        steps[num] = {
            "label": label,
            "text": text,
        }

    return steps


def steps_summary(steps: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    """
    Maak een korte samenvatting van geëxtraheerde stappen.

    Returns:
        Dict met: n_steps, n_expected (11), step_nums (gevonden stapnummers),
        total_chars (totaal aantal karakters over alle stappen).
    """
    return {
        "n_steps": len(steps),
        "n_expected": len(COT_STEPS),
        "step_nums": sorted(steps.keys(), key=lambda x: int(x)),
        "total_chars": sum(len(s["text"]) for s in steps.values()),
    }
