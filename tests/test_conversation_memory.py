from core.memory.conversation import roll_conversation
from core.memory.context import build_context
from core.task_store import TaskStore


def history(n):
    return [item for i in range(n) for item in (
        {'role': 'user', 'content': f'要求{i}，数值<{i + 1}。'},
        {'role': 'assistant', 'content': f'结果{i}。'})]


def test_recent_turns_are_verbatim_and_older_history_rolls_incrementally():
    calls = []
    def summarize(old, new):
        calls.append((old, new))
        return old + '；' + ','.join(item['content'] for item in new)
    first = roll_conversation(history(6), summarize=summarize)
    assert first['recent_messages'] == history(6)[4:]
    assert calls[0][1] == history(6)[:4]
    second = roll_conversation(history(7), first, summarize)
    assert calls[1] == (first['summary'], history(7)[4:6])
    assert second['covered_messages'] == 6
    roll_conversation(history(7), second, summarize)
    assert len(calls) == 2


def test_summary_persists_across_service_restart_and_is_task_scoped(tmp_path):
    tasks = TaskStore(tmp_path / 'tasks')
    task = tasks.create_task()['id']
    first = tasks.memory_service.conversation_context(task, history(6), lambda old, new: '历史摘要')
    restored = TaskStore(tmp_path / 'tasks').memory_service.conversation_context(task, history(6))
    assert restored == first
    other = tasks.create_task()['id']
    assert tasks.memory_service.conversation_context(other, history(6))['summary'] == ''


def test_failure_and_edited_history_do_not_silently_drop_messages(tmp_path):
    tasks = TaskStore(tmp_path / 'tasks')
    task = tasks.create_task()['id']
    def fail(*args):
        raise RuntimeError('unavailable')
    result = tasks.memory_service.conversation_context(task, history(6), fail)
    assert result['pending_messages'] + result['recent_messages'] == history(6)
    cached = roll_conversation(history(6), summarize=lambda old, new: '旧摘要')
    edited = history(6)
    edited[0]['content'] = '已修改'
    result = roll_conversation(edited, cached)
    assert result['summary'] == ''
    assert result['pending_messages'] == edited[:4]


def test_working_memory_keeps_recent_text_summary_and_constraints_even_over_budget():
    conversation = roll_conversation(history(6), summarize=lambda old, new: '先要求A，后撤销A')
    text, manifest = build_context({'conversation_memory': conversation,
                                  'task_memory': {'active_constraints': {'B': '有效规则B'}}}, budget=10)
    assert '先要求A，后撤销A' in text
    assert '有效规则B' in text
    for message in history(6)[4:]:
        assert message['content'] in text
    assert manifest['mandatory_over_budget']
    assert manifest['summary_covered_messages'] == 4


def test_real_provider_receives_rolling_summary(tmp_path):
    from PIL import Image
    from providers.vision import build_task_understanding_messages
    image = tmp_path / 'input.png'
    Image.new('RGB', (4, 4)).save(image)
    conversation = roll_conversation(history(6), summarize=lambda old, new: '用户撤销了旧要求')
    messages = build_task_understanding_messages(image, '继续', previous_context={'conversation_memory': conversation})
    text = '\n'.join(item.get('text', '') for item in messages[0]['content'])
    assert '用户撤销了旧要求' in text
    assert '要求5，数值<6。' in text


def test_provider_summary_request_is_text_only_and_bounded(monkeypatch):
    from types import SimpleNamespace
    import openai
    from providers.vision import AliyunVisionProvider
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='合并后的历史'))])
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    provider = AliyunVisionProvider(api_key='test-key')
    assert provider.summarize_conversation('旧摘要', history(1)) == '合并后的历史'
    request = requests[0]
    assert '4000' in request['messages'][0]['content']
    assert '旧摘要' in request['messages'][1]['content']
    assert 'tools' not in request


def test_oversized_summary_does_not_advance_saved_cursor(tmp_path):
    tasks = TaskStore(tmp_path / 'tasks')
    task = tasks.create_task()['id']
    before = tasks.memory_service.conversation_context(task, history(6), lambda old, new: '有效摘要')
    after = tasks.memory_service.conversation_context(task, history(7), lambda old, new: 'x' * 4001)
    assert after['summary'] == before['summary']
    assert after['covered_messages'] == before['covered_messages']
    assert after['pending_messages'] == history(7)[4:6]
