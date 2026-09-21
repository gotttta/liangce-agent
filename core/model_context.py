"""Measure each model request without logging prompt text or image payloads."""
import json
import os

from core.agent_events import emit_event


DEFAULT_MAX_TEXT_CHARS = 120000


class ContextBudgetError(ValueError):
    pass


def check_request_budget(messages, tools=None):
    """Character budget, not a tokenizer or image-token estimate.

    Fail explicitly if essential text still does not fit; never silently
    truncate requirements, source, failure causes, or visual evidence.
    """
    roles = {}
    image_count = image_url_chars = call_chars = 0
    for message in messages:
        text_chars = 0
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            if isinstance(part, str):
                text_chars += len(part)
            elif isinstance(part, dict) and part.get("type") == "image_url":
                image_count += 1
                image_url_chars += len(part.get("image_url", {}).get("url", ""))
            elif isinstance(part, dict):
                text_chars += len(str(part.get("text") or ""))
        arguments = len(json.dumps(message["tool_calls"], ensure_ascii=False)) if message.get("tool_calls") else 0
        call_chars += arguments
        role = message.get("role", "unknown")
        roles[role] = roles.get(role, 0) + text_chars + arguments
    schema_chars = len(json.dumps(tools, ensure_ascii=False)) if tools else 0
    limit = int(os.getenv("LIANGCE_LLM_MAX_TEXT_CHARS", str(DEFAULT_MAX_TEXT_CHARS)))
    if limit <= 0:
        raise ValueError("LIANGCE_LLM_MAX_TEXT_CHARS must be positive")
    total = sum(roles.values()) + schema_chars
    stats = {"message_count": len(messages), "text_chars_by_role": roles,
             "tool_call_chars": call_chars, "tool_schema_chars": schema_chars,
             "total_text_chars": total, "image_count": image_count,
             "image_url_chars": image_url_chars, "max_text_chars": limit,
             "over_budget": total > limit}
    emit_event({"type": "llm_input_size", **stats})
    image_limit = int(os.getenv('LIANGCE_LLM_MAX_IMAGES', '16'))
    byte_limit = int(os.getenv('LIANGCE_LLM_MAX_IMAGE_URL_CHARS', '24000000'))
    if image_count > image_limit or image_url_chars > byte_limit:
        raise ContextBudgetError('visual_context_budget_exceeded: reduce optional images or use scoped crops')
    if total > limit:
        raise ContextBudgetError(
            f"context_budget_exceeded: request contains {total} text characters, limit {limit}. "
            "Request was not sent. Requirements and evidence were not truncated; "
            "use scoped report pages or reduce optional context.")
    return stats


def compact_optional_history(messages, tools=None):
    """Offload resolved tool exchanges to a durable file when budgets overflow.

    Preserve the initial requirements and latest evidence. Tool calls and their
    replies are removed together so the provider protocol stays valid.
    """
    from copy import deepcopy
    from hashlib import sha256
    from pathlib import Path
    result = deepcopy(messages)
    text_limit = int(os.getenv('LIANGCE_LLM_MAX_TEXT_CHARS', str(DEFAULT_MAX_TEXT_CHARS)))
    image_limit = int(os.getenv('LIANGCE_LLM_MAX_IMAGES', '16'))
    def size():
        images = sum(1 for m in result if isinstance(m.get('content'), list)
                     for p in m['content'] if p.get('type') == 'image_url')
        text = json.dumps(result, ensure_ascii=False)
        # Image bytes have a separate bound.
        for m in result:
            if isinstance(m.get('content'), list):
                for p in m['content']:
                    if p.get('type') == 'image_url':
                        text = text.replace(p.get('image_url', {}).get('url', ''), '')
        return len(text) + len(json.dumps(tools or [])), images
    if size()[0] <= text_limit and size()[1] <= image_limit:
        return result
    starts = [i for i, m in enumerate(result) if m.get('tool_calls')]
    if len(starts) >= 2:
        first, last = starts[0], starts[-1]
        removed = result[first:last]
        raw = json.dumps(removed, ensure_ascii=False)
        directory = Path(os.getenv('LIANGCE_CONTEXT_ARCHIVE_DIR', 'workspace/context_archives'))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (sha256(raw.encode()).hexdigest() + '.json')
        path.write_text(raw, encoding='utf-8')
        result[first:last] = [{'role': 'user', 'content': (
            f'Earlier resolved tool exchanges were offloaded to {path}. Evidence is still available through inspect_experiment/query_operators; '
            'do not treat omitted evidence as passing. Re-read required evidence before acceptance.')}]
    return result
