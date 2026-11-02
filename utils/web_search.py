from __future__ import annotations

import os
import re
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


# ==========================================================
# Cache helpers
# ==========================================================
def _query_hash(query: str, max_results: int, fetch_top_n: int) -> str:
    raw = f"{query}|{max_results}|{fetch_top_n}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, query: str, max_results: int, fetch_top_n: int) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"web_{_query_hash(query, max_results, fetch_top_n)}.json")


def _load_cache(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================================================
# OpenAI Web Search (PRIMARY - via Responses API)
# ==========================================================
def build_openai_search_prompt(case_context: str) -> str:
    """Bouwt een Nederlandstalige zoekprompt op basis van de casecontext."""
    return (
        "Zoek op internet naar relevante informatie voor het volgende "
        "restaurantprobleem. Geef per gevonden bron een korte samenvatting "
        "van de relevante informatie. Vermeld altijd de titel en URL van "
        "elke bron.\n\n"
        f"CASE CONTEXT:\n{case_context}\n\n"
        "Zoek naar:\n"
        "1. Strategieen die aansluiten bij dit specifieke probleem\n"
        "2. Best practices uit de horeca-industrie\n"
        "3. Recente markttrends en concurrentie-inzichten\n"
    )


def _extract_openai_results(response) -> Tuple[List[Dict[str, Any]], str]:
    """Extraheer zoekresultaten (tekst + URL-citaties) uit een OpenAI Response."""
    results: List[Dict[str, Any]] = []
    full_text = ""

    for item in response.output:
        if item.type != "message":
            continue
        for block in item.content:
            if getattr(block, "type", None) != "output_text":
                continue
            full_text = block.text
            annotations = getattr(block, "annotations", []) or []
            seen_urls: set = set()
            rank = 0
            for ann in annotations:
                if getattr(ann, "type", None) != "url_citation":
                    continue
                url = getattr(ann, "url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                rank += 1
                results.append({
                    "rank": rank,
                    "title": getattr(ann, "title", ""),
                    "url": url,
                    "snippet": "",
                    "excerpt": "",
                })
    return results, full_text


def retrieve_web_context_openai(
    *,
    case_context: str,
    cache_dir: str,
    model: str = "gpt-4o-mini",
    max_results: int = 8,
    fetch_top_n: int = 5,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Haal webcontext op via OpenAI Responses API met web_search_preview.

    In tegenstelling tot DuckDuckGo werkt dit vanuit elke omgeving
    (serverless, cloud) omdat het een gewone API-call is.
    """
    from utils.openai_client import get_client

    search_prompt = build_openai_search_prompt(case_context)
    cp = _cache_path(cache_dir, search_prompt, max_results, fetch_top_n)

    if use_cache:
        cached = _load_cache(cp)
        if cached is not None:
            return cached

    client = get_client()

    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=search_prompt,
    )

    results, web_summary = _extract_openai_results(response)
    results = results[:fetch_top_n]

    for r in results:
        r["excerpt"] = ""

    usage = None
    if hasattr(response, "usage") and response.usage:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    payload = {
        "query": search_prompt[:200],
        "search_backend": "openai_web_search_preview",
        "model": model,
        "max_results": max_results,
        "fetch_top_n": fetch_top_n,
        "results": results,
        "web_summary": web_summary,
        "usage": usage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if use_cache:
        _save_cache(cp, payload)

    return payload


# ==========================================================
# DuckDuckGo Search (LEGACY fallback)
# ==========================================================
def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if DDGS is None:
        raise ImportError(
            "duckduckgo_search is niet geinstalleerd. "
            "Installeer met: pip install duckduckgo-search"
        )

    results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        rows = ddgs.text(query, max_results=max_results)
        for idx, row in enumerate(rows, start=1):
            results.append({
                "rank": idx,
                "title": row.get("title", ""),
                "url": row.get("href", ""),
                "snippet": row.get("body", ""),
            })
    return results


def fetch_page_excerpt(url: str, timeout: int = 10, max_chars: int = 1500) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
    except Exception:
        return ""
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return ""
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return ""
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = clean_text(text)
    if not text:
        return ""
    return text[:max_chars]


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def build_search_query(case_context: str, query_suffix: str = "") -> str:
    """Bouwt een abstracte zoekquery voor DuckDuckGo (legacy)."""
    text = clean_text(case_context).lower()
    terms: List[str] = ["restaurant", "hospitality", "strategy"]
    if _contains_any(text, ["student", "studenten"]):
        terms.extend(["student market", "demand generation"])
    if _contains_any(text, ["jonge gezinnen", "gezinnen", "familie", "kinderen"]):
        terms.extend(["family dining", "target market fit"])
    if _contains_any(text, ["oudere doelgroep", "oudere"]):
        terms.extend(["customer segmentation", "repositioning"])
    if _contains_any(text, ["omzet daalt", "minder gasten", "laag blijft", "reserveringen laag", "daalt"]):
        terms.extend(["low demand", "revenue improvement"])
    if _contains_any(text, ["doordeweekse avonden", "weekdagen"]):
        terms.extend(["weekday traffic", "off-peak demand"])
    if _contains_any(text, ["weekend"]):
        terms.extend(["weekend demand"])
    if _contains_any(text, ["zomer", "winter", "toeristische", "kustplaats", "seizoen"]):
        terms.extend(["seasonality", "tourism demand", "off-season strategy"])
    if _contains_any(text, ["concurrent", "nieuw concept", "zelfde straat", "lagere prijzen"]):
        terms.extend(["competitive positioning", "restaurant differentiation", "pricing strategy"])
    if _contains_any(text, ["fine-dining", "duur", "dure", "traditioneel menu", "menu"]):
        terms.extend(["menu engineering", "menu repositioning", "pricing"])
    if _contains_any(text, ["modern concept"]):
        terms.extend(["concept innovation", "restaurant positioning"])
    if _contains_any(text, ["vaste kosten", "winst daalt", "personeelskosten"]):
        terms.extend(["cost control", "profitability"])
    if _contains_any(text, ["annuleringen", "last-minute annuleringen"]):
        terms.extend(["reservation policy", "no-show reduction"])
    if _contains_any(text, ["goede reputatie", "veel vraag naar tafels"]):
        terms.extend(["capacity management", "yield management"])
    seen = set()
    deduped_terms = []
    for term in terms:
        if term not in seen:
            deduped_terms.append(term)
            seen.add(term)
    if query_suffix:
        deduped_terms.append(query_suffix.strip())
    return " ".join(deduped_terms).strip()


def is_reasonably_relevant_result(item: Dict[str, Any]) -> bool:
    title = (item.get("title") or "").lower()
    url = (item.get("url") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    bad_markers = [
        "zhihu", "quizlet", "reddit", "pinterest",
        "restaurant pronunciation", "at a restaurant",
        "in a restaurant", "conversation questions",
        "translation", "dictionary", "lyrics",
    ]
    haystack = f"{title} {url} {snippet}"
    return not any(marker in haystack for marker in bad_markers)


def retrieve_web_context(
    *,
    query: str,
    cache_dir: str,
    max_results: int = 8,
    fetch_top_n: int = 3,
    use_cache: bool = True,
    request_sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Legacy DuckDuckGo pipeline. Werkt NIET vanuit Databricks serverless."""
    cp = _cache_path(cache_dir, query, max_results, fetch_top_n)
    if use_cache:
        cached = _load_cache(cp)
        if cached is not None:
            return cached
    raw_results = search_web(query=query, max_results=max_results)
    filtered_results = [r for r in raw_results if is_reasonably_relevant_result(r)]
    enriched_results: List[Dict[str, Any]] = []
    for row in filtered_results[:fetch_top_n]:
        excerpt = fetch_page_excerpt(row["url"])
        enriched_results.append({**row, "excerpt": excerpt})
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)
    payload = {
        "query": query,
        "search_backend": "duckduckgo",
        "max_results": max_results,
        "fetch_top_n": fetch_top_n,
        "results": enriched_results,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if use_cache:
        _save_cache(cp, payload)
    return payload


# ==========================================================
# Context builder (werkt met BEIDE backends)
# ==========================================================
def build_augmented_web_context(
    case_context: str,
    web_payload: Dict[str, Any],
) -> str:
    """Bouwt de augmented context string voor de LLM prompt."""
    results = web_payload.get("results", [])
    web_summary = web_payload.get("web_summary", "")

    if not results and not web_summary:
        web_text = "Geen aanvullende internetinformatie gevonden."
    elif web_summary:
        source_list = "\n".join(
            f"  [{i}] {r.get('title', '')} - {r.get('url', '')}"
            for i, r in enumerate(results, start=1)
        )
        web_text = f"{web_summary}\n\nGebruikte bronnen:\n{source_list}"
    else:
        blocks = []
        for i, item in enumerate(results, start=1):
            block = (
                f"[Webbron {i}]\n"
                f"Titel: {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Snippet: {item.get('snippet', '')}\n"
                f"Excerpt: {item.get('excerpt', '')}"
            )
            blocks.append(block)
        web_text = "\n\n".join(blocks)

    return (
        "CASE_CONTEXT:\n"
        f"{case_context.strip()}\n\n"
        "AANVULLENDE_INTERNET_CONTEXT:\n"
        f"{web_text}"
    )
