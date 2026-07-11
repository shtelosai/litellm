"""Twork: roundtrip OpenAI Responses reasoning items over the Anthropic protocol.

Reasoning-heavy Responses models (GPT 5.x Codex family) emit encrypted reasoning
items each turn. Official agent loops (Codex CLI) send those items back on the
next request; without them the model loses its own chain-of-thought every round
and degrades badly in agentic tool loops (premature end_turn, unconsumed tool
results). Upstream tracking: https://github.com/BerriAI/litellm/issues/24425

This module packs reasoning items into the ``data`` field of a synthetic
Anthropic ``redacted_thinking`` content block (prefix-marked), so any Claude-protocol
client that echoes assistant content verbatim transparently roundtrips them.
All wiring lives in Anthropic-adapter-only code paths: plain
``/chat/completions`` clients can never observe the synthetic blocks.

Kill switch: set ``LITELLM_TWORK_REASONING_ROUNDTRIP=0`` (default: enabled).
"""

import base64
import binascii
import json
import os
from typing import Any, Dict, List, Optional

SIGNATURE_PREFIX = "tworkrs1:"

_ENV_FLAG = "LITELLM_TWORK_REASONING_ROUNDTRIP"
_ENCRYPTED_CONTENT_INCLUDE = "reasoning.encrypted_content"


def is_enabled() -> bool:
    """Feature flag, default on. ``LITELLM_TWORK_REASONING_ROUNDTRIP=0`` disables."""
    return os.getenv(_ENV_FLAG, "1") != "0"


def with_encrypted_content_include(params: Dict[str, Any]) -> Dict[str, Any]:
    if not is_enabled():
        return params
    include = params.get("include")
    existing = include if isinstance(include, list) else []
    if _ENCRYPTED_CONTENT_INCLUDE in existing:
        return params
    return {**params, "include": [*existing, _ENCRYPTED_CONTENT_INCLUDE]}


def _get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def pack_reasoning_items(items: Any) -> Optional[str]:
    """Pack reasoning items carrying ``encrypted_content`` into a marked data string.

    Returns ``None`` when there is nothing worth roundtripping (no items or no
    encrypted payloads).
    """
    if not items:
        return None
    packed: List[Dict[str, Any]] = []
    for item in items:
        encrypted_content = _get(item, "encrypted_content")
        if not encrypted_content:
            continue
        packed.append({"id": _get(item, "id") or "", "ec": encrypted_content})
    if not packed:
        return None
    return SIGNATURE_PREFIX + base64.b64encode(json.dumps(packed).encode()).decode()


def unpack_signature(signature: Any) -> Optional[List[Dict[str, Any]]]:
    """Decode a prefix-marked data string back into reasoning items.

    Returns ``None`` for non-marked or malformed signatures (never raises).
    """
    if not isinstance(signature, str) or not signature.startswith(SIGNATURE_PREFIX):
        return None
    try:
        payload = json.loads(base64.b64decode(signature[len(SIGNATURE_PREFIX) :]).decode())
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    items: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict) or not entry.get("ec"):
            continue
        items.append(
            {
                "id": entry.get("id") or "",
                "type": "reasoning",
                "encrypted_content": entry["ec"],
                "summary": [],
            }
        )
    return items or None


def summary_text(items: Any) -> str:
    """Join reasoning item summary texts for display inside the thinking block."""
    texts: List[str] = []
    for item in items or []:
        for entry in _get(item, "summary") or []:
            text = entry.get("text", "") if isinstance(entry, dict) else getattr(entry, "text", "")
            if text:
                texts.append(text)
    return " ".join(texts)
