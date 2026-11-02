from __future__ import annotations
from openai import OpenAI
from configs.config import load_openai_config

def get_client() -> OpenAI:
    cfg = load_openai_config()
    if not cfg.api_key or cfg.api_key.startswith("PASTE_"):
        raise RuntimeError(
            "OpenAI API key ontbreekt. Zet OPENAI_API_KEY in je environment "
            "of vul configs/config.py aan."
        )
    kwargs = {"api_key": cfg.api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)
