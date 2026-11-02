"""
Single source of truth prompt for all experiments (herhaalbaarheid).

Gebruik:
from configs.prompts import build_messages, PROMPT_ID, QUESTION
"""

from __future__ import annotations
import hashlib
from typing import Optional, List, Dict


# ==========================================================
# VASTE ONDERZOEKSVRAAG
# ==========================================================

QUESTION = (
    "Ontwikkel een strategisch advies voor dit restaurant op basis van de gegeven context."
)


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_INSTRUCTIONS = (
    "Je bent een behulpzame assistent die strategisch advies geeft aan een restaurant.\n"
    "Je werkt op academisch niveau en redeneert gestructureerd en expliciet.\n\n"

    "Je volgt principes uit evidence-based management (Rousseau, 2006):\n"
    "- Adviezen zijn gebaseerd op analyse\n"
    "- Redeneringen zijn expliciet en navolgbaar\n\n"

    "Je werkt volgens een ontwerpgerichte aanpak (Van Aken et al., 2012):\n"
    "- Aannames expliciet maken\n"
    "- Transparant redeneren\n\n"

    "Je hanteert principes uit strategisch management:\n"
    "- Externe analyse van markt, concurrentie en trends\n"
    "- Interne analyse van middelen, capaciteiten en prestaties\n"
    "- Strategie als samenhangend geheel van keuzes\n\n"

    "Je gebruikt data uit de context als basis voor analyse, in lijn met "
    "Decision Support Systems-principes.\n\n"

    "Regels:\n"
    "- Gebruik uitsluitend informatie uit de gegeven CONTEXT.\n"
    "- Als iets niet in de context staat, zeg: "
    "'Ik weet het niet op basis van de gegeven context.'\n"
    "- Maak aannames expliciet als informatie ontbreekt.\n"
    "- Werk logisch van analyse naar conclusie "
    "(oorzaak → gevolg → implicatie → advies).\n"
    "- Vermijd vage of algemene formuleringen.\n"
    "- Focus op strategische relevantie.\n"
)


# ==========================================================
# USER TEMPLATE
# ==========================================================

USER_TEMPLATE = (
    "CONTEXT:\n"
    "{context}\n\n"

    "VRAAG:\n"
    "{question}\n\n"

    "INSTRUCTIE:\n"
    "Ontwikkel een strategisch advies voor dit restaurant op basis van analyse "
    "en, waar relevant, scenario-denken.\n\n"

    "Werk volgens deze structuur:\n\n"

    "1. Probleemdefinitie en doel\n"
    "- Beschrijf de huidige situatie van het restaurant.\n"
    "- Definieer het strategisch vraagstuk.\n"
    "- Leg uit waarom dit strategisch relevant is.\n\n"

    "2. Externe analyse\n"
    "- Analyseer markt, concurrentie en relevante trends.\n"
    "- Identificeer kansen en bedreigingen.\n"
    "- Maak onderscheid tussen relatief zekere ontwikkelingen en onzekerheden.\n\n"

    "3. Interne analyse\n"
    "- Analyseer prestaties, middelen en capaciteiten.\n"
    "- Gebruik data uit de context (bijv. menu, prijzen, kosten, doelgroep, operatie).\n"
    "- Identificeer sterktes en zwaktes.\n\n"

    "4. Synthese van interne en externe analyse\n"
    "- Combineer de inzichten uit de externe en interne analyse.\n"
    "- Leg expliciet verbanden tussen sterktes, zwaktes, kansen en bedreigingen.\n"
    "- Benoem de belangrijkste strategische spanningen of kansen.\n\n"

    "5. Scenario-analyse (indien relevant)\n"
    "- Identificeer de belangrijkste onzekerheden.\n"
    "- Werk 2 tot 4 korte plausibele scenario's uit als dit helpt om het advies te onderbouwen.\n"
    "- Gebruik scenario's alleen als ondersteunend instrument voor strategische keuze.\n\n"

    "6. Strategische opties\n"
    "- Ontwikkel 2 tot 3 mogelijke strategische richtingen.\n"
    "- Beschrijf per optie:\n"
    "  - wat er verandert\n"
    "  - waarom deze optie logisch is\n"
    "  - de verwachte impact\n"
    "  - mogelijke risico's of nadelen\n\n"

    "7. Evaluatie en keuze\n"
    "- Vergelijk de strategische opties op haalbaarheid en verwachte impact.\n"
    "- Kies één voorkeursstrategie.\n"
    "- Leg expliciet uit waarom deze keuze het best past bij de context.\n\n"

    "8. Strategisch advies\n"
    "- Formuleer één duidelijke strategische richting.\n"
    "- Onderbouw deze expliciet op basis van de analyse.\n"
    "- Benoem waar relevant robuuste of no-regret maatregelen.\n\n"

    "9. Implementatie\n"
    "- Benoem 2 tot 4 concrete acties.\n"
    "- Geef prioriteit aan de acties.\n"
    "- Licht kort toe waarom deze volgorde logisch is.\n\n"

    "10. Monitoring en bijsturing\n"
    "- Benoem hoe de voortgang of prestaties gemeten kunnen worden.\n"
    "- Geef 2 tot 3 relevante KPI's of signalen.\n"
    "- Leg uit wanneer bijsturing nodig zou zijn.\n\n"

    "11. Aannames en beperkingen\n"
    "- Benoem expliciet welke aannames zijn gemaakt.\n"
    "- Benoem welke informatie ontbreekt.\n"
    "- Geef aan hoe deze beperkingen het advies beïnvloeden.\n\n"

    "BELANGRIJK:\n"
    "- Gebruik alleen informatie uit de context.\n"
    "- Maak redeneringen expliciet (omdat → daarom → dus).\n"
    "- Focus op strategische relevantie.\n"
    "- Vermijd herhaling.\n"
    "- Geef een concreet, onderbouwd en uitvoerbaar advies.\n"
)


# ==========================================================
# MESSAGE BUILDER
# ==========================================================

def build_messages(context: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Bouwt de volledige prompt.

    Question komt uit deze file zodat alle experimenten
    exact dezelfde vraag gebruiken.
    """
    if context is None:
        context = ""

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                context=context,
                question=QUESTION,
            ),
        },
    ]


# ==========================================================
# PROMPT HASH (voor reproduceerbaarheid)
# ==========================================================

def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


PROMPT_ID = (
    f"sys:{_hash_text(SYSTEM_INSTRUCTIONS)}"
    f"|usr:{_hash_text(USER_TEMPLATE)}"
    f"|q:{_hash_text(QUESTION)}"
)