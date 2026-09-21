import json
import os
import subprocess
import sys

from core.task_store import TaskStore
from core.memory.context import build_context
from core.memory.service import image_hash


def service(tmp_path):
    tasks = TaskStore(tmp_path / 'tasks')
    task = tasks.create_task()
    return tasks, tasks.memory_service, task['id']


def test_constraints_survive_revoke_is_versioned_and_isolated(tmp_path):
    tasks, memory, task_id = service(tmp_path)
    memory.set_fact(task_id, 'constraint:small', '保留小黑点', source_id='message:1')
    memory.apply_updates(task_id, '继续', {'target_constraints': {'max_coverage': .2}}, 'message:2')
    assert memory.snapshot(task_id)['active_constraints'] == {'small': '保留小黑点'}
    assert memory.snapshot(tasks.create_task()['id'])['active_constraints'] == {}
    memory.apply_updates(task_id, '取消小黑点要求', {'memory_updates': [
        {'op': 'revoke', 'key': 'constraint:small', 'source_quote': '取消小黑点要求'}]}, 'message:3')
    assert memory.snapshot(task_id)['active_constraints'] == {}
    versions = [r for r in memory.store.list(task_id, 'semantic', history=True) if r['key'] == 'constraint:small']
    assert [r['status'] for r in versions] == ['superseded', 'revoked']
    assert tasks.memory_service.snapshot(task_id)['active_constraints'] == {}


def test_updates_require_actual_user_source_and_goal_can_change(tmp_path):
    _, memory, task_id = service(tmp_path)
    memory.prepare(task_id, '提取颗粒', None)
    memory.apply_updates(task_id, '改为只检测划痕', {'memory_updates': [
        {'op': 'set', 'key': 'current_goal', 'value': '检测划痕', 'source_quote': '改为只检测划痕'},
        {'op': 'set', 'key': 'constraint:invented', 'value': 'exactly 5', 'source_quote': '不存在的原文'}
    ], 'target_constraints': {'expected_count': 5}}, 'message:2')
    snap = memory.snapshot(task_id)
    assert snap['current_goal'] == '检测划痕'
    assert snap['active_constraints'] == {}
    assert snap['semantic_hypotheses'][0]['status'] == 'hypothesis'


def test_legacy_memory_is_not_promoted_to_confirmed_fact(tmp_path):
    tasks, memory, task_id = service(tmp_path)
    tasks.save_memory(task_id, {'task_goal': '检测颗粒', 'active_constraints': {'expected_count': 7}})
    snap = memory.snapshot(task_id)
    assert snap['active_constraints'] == {}
    assert len(snap['semantic_hypotheses']) == 2
    memory.snapshot(task_id)
    assert len(memory.store.list(task_id, 'semantic', history=True)) == 2


def test_new_image_discards_pixel_feedback_and_old_baseline(tmp_path):
    _, memory, task_id = service(tmp_path)
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_bytes(b'a'); b.write_bytes(b'b')
    prior = {'target_image_path': str(a), 'input_sha256': image_hash(a),
             'include_mask_path': 'mask', 'pipeline': {'name': 'old'},
             'human_feedback': {'exclude_mask_path': 'old-mask'}}
    previous, context = memory.prepare(task_id, '检测', b, prior)
    assert previous is None
    assert context['human_feedback'] == {}
    assert context['previous_pipeline'] is None


def test_prompt_keeps_constraints_and_strips_ui_html_under_budget():
    text, manifest = build_context({'task_memory': {'current_goal': '颗粒',
        'active_constraints': {'required': '不能丢' * 100},
        'constraint_records': [{'id': 'fact1'}]},
        'conversation': [{'role': 'assistant', 'content': '<div class="progress-card">noise</div>'}],
        'previous_pipeline': {'source': 'x' * 5000}}, budget=200)
    assert '不能丢' * 100 in text
    assert 'noise' not in text
    assert manifest['memory_ids'] == ['fact1']
    assert manifest['mandatory_over_budget']
    assert 'previous_pipeline' in manifest['omitted']


def test_real_provider_prompt_and_review_receive_memory(tmp_path):
    from PIL import Image
    from providers.vision import build_task_understanding_messages, build_candidate_review_messages
    image = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(image)
    contract = {'current_goal': '检测划痕', 'active_constraints': {'edge': '边缘必须包含'}}
    messages = build_task_understanding_messages(image, '继续', previous_context={
        'task_memory': contract, 'conversation': [{'role': 'user', 'content': '不要遗漏左上角'}]})
    text = json.dumps(messages, ensure_ascii=False)
    assert '边缘必须包含' in text and '不要遗漏左上角' in text
    review = build_candidate_review_messages(image, '继续', [], acceptance_criteria={'memory_contract': contract})
    assert '边缘必须包含' in json.dumps(review, ensure_ascii=False)


def test_checkpoint_survives_process_restart_without_reexecuting(tmp_path):
    script = '''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from core.memory.checkpoints import get_checkpointer
from pathlib import Path
import sys
class State(TypedDict):
    n: int
    answer: str
def execute(state):
    with Path(sys.argv[2]).open('a') as f: f.write('executed\\n')
    return {'n': state['n'] + 1}
def review(state):
    return {'answer': interrupt('accept?')}
b = StateGraph(State)
b.add_node('execute', execute); b.add_node('review', review)
b.add_edge(START, 'execute'); b.add_edge('execute', 'review'); b.add_edge('review', END)
g = b.compile(checkpointer=get_checkpointer())
x = g.invoke({'n': 0} if sys.argv[1] == 'start' else Command(resume='accepted'),
             {'configurable': {'thread_id': 'persisted'}})
assert x['n'] == 1
if sys.argv[1] != 'start': assert x['answer'] == 'accepted'
'''
    env = {**os.environ, 'LIANGCE_CHECKPOINT_PATH': str(tmp_path / 'checkpoints.sqlite3')}
    counter = tmp_path / 'executions'
    for action in ('start', 'resume'):
        subprocess.run([sys.executable, '-c', script, action, str(counter)], env=env, check=True, capture_output=True)
    assert counter.read_text() == 'executed\n'


def test_graph_entrypoint_reuses_paused_thread_and_keeps_task_memory(tmp_path, monkeypatch):
    from PIL import Image
    from core.agent_graph import run_agent_graph, resume_agent_graph
    import core.agent_graph as graph_module
    monkeypatch.setenv('LIANGCE_CHECKPOINT_PATH', str(tmp_path / 'graph.sqlite3'))
    image = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(image)
    tasks, memory, task_id = service(tmp_path)
    calls = []
    def execute(state):
        calls.append(1)
        return {**state, 'candidate_attempts': [], 'quality_report': {},
                'measurements': {}, 'iteration': 0, 'status': 'completed'}
    monkeypatch.setattr(graph_module, '_execute_candidates', execute)
    understanding = {'task_summary': '检测', 'recommended_strategy': {}, 'target_constraints': {},
                     'candidate_pipelines': [], 'memory_updates': [
                         {'op': 'set', 'key': 'constraint:edge', 'value': '包含边缘', 'source_quote': '包含边缘'}]}
    first = run_agent_graph(image, '包含边缘', output_root=tmp_path / 'outputs',
                            task_store=tasks, task_id=task_id, thread_id='test-run', understanding=understanding,
                            max_auto_revisions=0)
    assert first['task_id'] == task_id
    assert first['memory_summary']['active_constraints']['edge'] == '包含边缘'
    episodes_before = len(memory.store.list(task_id, 'episodic'))
    repeated = run_agent_graph(image, '包含边缘', output_root=tmp_path / 'outputs',
                               task_store=tasks, task_id=task_id, thread_id='test-run', understanding=understanding)
    assert repeated['interrupt']
    assert len(calls) == 1
    assert len(memory.store.list(task_id, 'episodic')) == episodes_before
    resumed = resume_agent_graph('test-run', {'action': 'accept'})
    assert resumed['agent_status'] == 'accepted'
    assert resume_agent_graph('test-run', {'action': 'accept'})['agent_status'] == 'accepted'
    assert len(calls) == 1


def test_accepted_algorithm_creates_procedural_evidence(tmp_path):
    tasks, memory, task_id = service(tmp_path)
    acceptance = tasks.accept_result(task_id, {
        'pipeline': {'name': 'accepted', 'steps': []},
        'strategy': {'defect_type': 'particle'}, 'description': '检测颗粒'})
    records = memory.store.list('algorithms', 'procedural')
    assert records[0]['data']['algorithm_id'] == acceptance['registry_algorithm_id']
    assert records[0]['data']['validation'] == 'user_accepted_on_source_input'
    assert records[0]['data']['source_task_id'] == task_id
