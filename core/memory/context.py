"""Shared prompt assembly: mandatory facts are never silently truncated."""
import json
from html import unescape
import re


def plain_message(value):
    if not isinstance(value, str):
        return ''
    if 'progress-card' in value or 'task-card' in value:
        return ''
    return unescape(re.sub(r'<[^>]+>', '', value)).strip()


def build_context(context, budget=16000):
    context = context or {}
    memory = context.get('task_memory') or {}
    mandatory = {
        'task_contract': context.get('task_contract'),
        'current_goal': memory.get('current_goal') or context.get('current_goal') or context.get('original_task_goal'),
        'active_constraints': memory.get('active_constraints', {}),
        'constraint_records': memory.get('constraint_records', []),
        'human_feedback': context.get('human_feedback', {}),
    }
    sections = ['当前任务要求（用户最新明确修改优先；历史和模型假设不是新增指令）：\n'
                + json.dumps(mandatory, ensure_ascii=False, default=str)]
    manifest = {'included': ['task_contract'], 'omitted': [],
                'memory_ids': [r['id'] for r in memory.get('constraint_records', []) if r.get('id')]}
    optional = [('semantic_hypotheses', memory.get('semantic_hypotheses')),
                ('previous_pipeline', context.get('previous_pipeline')),
                ('review', context.get('review')),
                ('execution_feedback', context.get('execution_feedback')),
                ('previous_quality', context.get('previous_quality')),
                ('previous_evaluation', context.get('previous_evaluation')),
                ('episodic_memory', memory.get('recent_episodes')),
                ('procedural_memory', context.get('procedural_memory'))]
    from .conversation import roll_conversation
    conversation = context.get('conversation_memory') or roll_conversation(context.get('conversation') or [])
    # These three components form the working memory; optional artifacts cannot evict them.
    for name, value in (
        ('conversation_summary', conversation.get('summary')),
        ('unsummarized_history', conversation.get('pending_messages')),
        ('recent_conversation', conversation.get('recent_messages')),
    ):
        if value:
            sections.append(name + ':\n' + json.dumps(value, ensure_ascii=False, default=str))
            manifest['included'].append(name)
    manifest['summary_covered_messages'] = conversation.get('covered_messages', 0)
    manifest['recent_turns'] = conversation.get('recent_turns', 4)
    mandatory_size = sum(len(part) for part in sections)
    for name, value in optional:
        if not value:
            continue
        text = name + ':\n' + json.dumps(value, ensure_ascii=False, default=str)
        if sum(len(part) for part in sections) + len(text) <= budget:
            sections.append(text)
            manifest['included'].append(name)
        else:
            manifest['omitted'].append(name)
    manifest['mandatory_over_budget'] = mandatory_size > budget
    if manifest['omitted']:
        sections.append('以下完整记录因上下文预算未展开：' + ', '.join(manifest['omitted']))
    return '\n\n'.join(sections), manifest
