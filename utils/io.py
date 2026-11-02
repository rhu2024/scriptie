from __future__ import annotations
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================================
# Structured JSON → flat context string
# ==========================================================

def _format_key_value(key: str, value: Any, indent: int = 0) -> str:
    """Formatteert een key-value paar als leesbare tekst."""
    label = key.replace("_", " ").capitalize()
    prefix = "  " * indent

    if isinstance(value, str):
        return f"{prefix}{label}: {value}"
    if isinstance(value, (int, float)):
        return f"{prefix}{label}: {value}"
    if isinstance(value, list):
        return _format_list_section(label, value, indent)
    if isinstance(value, dict):
        lines = [f"{prefix}{label}:"]
        for k, v in value.items():
            lines.append(_format_key_value(k, v, indent + 1))
        return "\n".join(lines)
    return f"{prefix}{label}: {value}"


def _format_list_section(label: str, items: List[Any], indent: int = 0) -> str:
    """Formatteert een lijst van menu-items of strings."""
    prefix = "  " * indent
    lines = [f"{prefix}{label}:"]

    for item in items:
        if isinstance(item, dict):
            naam = item.get("naam", "")
            prijs = item.get("prijs", "")
            beschrijving = item.get("beschrijving", "")
            line = f"{prefix}  - {naam} | {prijs}"
            if beschrijving:
                line += f"\n{prefix}    {beschrijving}"
            lines.append(line)
        elif isinstance(item, str):
            lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}  - {item}")

    return "\n".join(lines)


def _build_context_from_structured(data: Dict[str, Any]) -> str:
    """
    Bouwt een platte context-string op uit gestructureerde case-velden.
    Herkent: restaurantprofiel, positionering, doelgroep, marktcontext, menukaart.
    """
    sections: List[str] = []

    section_order = [
        ("restaurantprofiel", "RESTAURANTPROFIEL"),
        ("positionering", "POSITIONERING"),
        ("doelgroep", "DOELGROEP"),
        ("marktcontext", "MARKTCONTEXT"),
        ("menukaart", "MENUKAART"),
    ]

    for key, header in section_order:
        if key not in data:
            continue

        value = data[key]

        if isinstance(value, str):
            sections.append(f"{header}\n{value}")

        elif isinstance(value, dict) and key == "menukaart":
            menu_lines = [header]
            for categorie, items in value.items():
                cat_label = categorie.replace("_", " ").upper()
                menu_lines.append(f"\n{cat_label}")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            naam = item.get("naam", "")
                            prijs = item.get("prijs", "")
                            beschrijving = item.get("beschrijving", "")
                            menu_lines.append(f"- {naam} | {prijs}")
                            if beschrijving:
                                menu_lines.append(f"  {beschrijving}")
                        else:
                            menu_lines.append(f"- {item}")
            sections.append("\n".join(menu_lines))

        elif isinstance(value, dict):
            lines = [header]
            for k, v in value.items():
                lines.append(_format_key_value(k, v))
            sections.append("\n".join(lines))

        else:
            sections.append(f"{header}\n{value}")

    return "\n\n".join(sections)


# ==========================================================
# Case file reader
# ==========================================================

def read_case_file(path: str) -> Dict[str, Any]:
    """
    Ondersteunt:
    - .json met gestructureerde velden (restaurantprofiel, menukaart, etc.)
      → worden automatisch omgezet naar een platte context-string
    - .json met een direct 'context' veld (legacy formaat)
    - .txt / .md => hele inhoud wordt context
    """
    _, ext = os.path.splitext(path.lower())

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Case JSON moet een object zijn: {path}")

        data.setdefault("metadata", {})
        data.setdefault("question", None)

        # Gestructureerd formaat: bouw context uit losse secties
        if "context" not in data and "restaurantprofiel" in data:
            data["context"] = _build_context_from_structured(data)
        else:
            data.setdefault("context", "")

        return data

    elif ext in (".txt", ".md"):
        return {"context": read_text(path), "question": None, "metadata": {}}

    else:
        raise ValueError(f"Onbekend case file type: {path}")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
