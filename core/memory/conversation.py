"""Rolling conversation summaries, with an exact recent-turn window."""
import hashlib
import json

RECENT_TURNS = 4
SUMMARY_CHARS = 4000
BATCH_CHARS = 16000


def conversation_messages(messages):
    """Keep actual text verbatim; rendered UI cards and file payloads are not prose."""
    result = []
    for item in messages or []:
        if not isinstance(item, dict) or item.get('role') not in {'user', 'assistant'}:
            continue
        content = item.get('content')
        if not isinstance(content, str) or not content.strip():
            continue
        if 'progress-card' in content or 'task-card' in content:
            # An assistant reply may have prose followed by a rendered card.
            import re
            content = re.split(r'<(?:div|section)\b[^>]*class=["\'][^"\']*(?:progress-card|task-card)', content, maxsplit=1)[0].rstrip()
            if not content or 'progress-card' in content or 'task-card' in content:
                continue
        result.append({'role': item['role'], 'content': content})
    return result


def split_turns(messages, recent_turns=RECENT_TURNS):
    if recent_turns < 1:
        raise ValueError('recent_turns must be positive')
    messages = conversation_messages(messages)
    starts = [i for i, item in enumerate(messages) if item['role'] == 'user']
    boundary = starts[-recent_turns] if len(starts) > recent_turns else 0
    return messages[:boundary], messages[boundary:]


def fingerprint(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def roll_conversation(messages, cached=None, summarize=None, recent_turns=RECENT_TURNS):
    older, recent = split_turns(messages, recent_turns)
    cached = cached or {}
    count = cached.get('covered_messages', 0)
    if (not isinstance(count, int) or count < 0 or count > len(older)
            or cached.get('prefix_hash') != fingerprint(older[:count])):
        cached, count = {}, 0
    summary = cached.get('summary', '')
    pending = older[count:]
    if summarize:
        while pending:
            batch, size = [], 0
            for item in pending:
                length = len(item['content'])
                if batch and size + length > BATCH_CHARS:
                    break
                batch.append(item)
                size += length
            if len(batch) == 1 and len(batch[0]['content']) > BATCH_CHARS:
                item = batch[0]
                for start in range(0, len(item['content']), BATCH_CHARS):
                    updated = summarize(summary, [{**item, 'content': item['content'][start:start + BATCH_CHARS]}])
                    if not isinstance(updated, str) or not updated.strip() or len(updated) > SUMMARY_CHARS:
                        raise ValueError('Conversation summary must be nonempty and within its budget')
                    summary = updated.strip()
                count += 1
                pending = older[count:]
                continue
            # No source slice is lost on failure. The caller can retry next turn.
            updated = summarize(summary, batch)
            if not isinstance(updated, str) or not updated.strip() or len(updated) > SUMMARY_CHARS:
                raise ValueError('Conversation summary must be nonempty and within its budget')
            summary = updated.strip()
            count += len(batch)
            pending = older[count:]
    return {
        'summary': summary, 'covered_messages': count,
        'prefix_hash': fingerprint(older[:count]), 'recent_turns': recent_turns,
        'recent_messages': recent, 'pending_messages': pending,
    }
