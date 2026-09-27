"""One Agent decision per request, using the existing tool contracts."""
from copy import deepcopy
import json

from core.tools.contracts import TOOL_SPECS


AGENT_TOOLS = (
    'query_operators', 'load_skill', 'read_draft', 'inspect_experiment', 'inspect_artifact',
    'create_draft', 'edit_draft', 'execute_pipeline', 'compare_candidates', 'submit_experiment',
)


def tool_schemas():
    schemas = [deepcopy(TOOL_SPECS[name].function_schema()['function']) for name in AGENT_TOOLS]
    for spec in schemas:
        parameters = spec['parameters']
        if spec['name'] in {'create_draft', 'edit_draft'}:
            parameters['properties'].update(
                contract_updates={'type': 'array', 'items': {'type': 'object'}},
                memory_updates={'type': 'array', 'items': {'type': 'object'}})
        if spec['name'] == 'create_draft':
            parameters['properties'].pop('parent_experiment_id', None)
            parameters['properties']['understanding'] = {'type': 'object'}
        if spec['name'] == 'execute_pipeline':
            # Persisted, validated drafts are the only executable input in this graph.
            parameters['properties'] = {key: parameters['properties'][key] for key in ('draft_id', 'revision')}
            parameters['required'] = ['draft_id', 'revision']
    return schemas


def normalize_agent_action(raw, *, description, context=None):
    from providers.vision import normalize_model_action
    from core.tools.contracts import _validate
    if isinstance(raw, dict) and raw.get('kind') == 'needs_input':
        return normalize_model_action(raw, description=description, context=context)
    if not isinstance(raw, dict) or set(raw) != {'kind', 'tool', 'arguments'} or raw.get('kind') != 'tool':
        raise ValueError('Return one {kind:tool, tool, arguments} action, or needs_input')
    name, arguments = raw['tool'], raw['arguments']
    schema = next((item for item in tool_schemas() if item['name'] == name), None)
    if schema is None:
        raise ValueError('Unknown Agent tool')
    _validate(arguments, schema['parameters'], 'arguments')
    arguments = deepcopy(arguments)
    if name in {'create_draft', 'edit_draft'}:
        proposal = normalize_model_action(
            {'kind': 'propose' if name == 'create_draft' else 'edit', **arguments},
            description=description, context=context)
        arguments = {key: value for key, value in proposal.items() if key != 'kind'}
    return {'kind': 'tool', 'tool': name, 'arguments': arguments}


def build_agent_messages(target_image_path, description, *, context=None, reference_examples=None):
    from providers.vision import build_action_messages
    messages = build_action_messages(target_image_path, description, context=context,
                                     reference_examples=reference_examples)
    system = messages[0]['content']
    # Share image/DSL/contract guidance with the compatibility provider, but replace
    # its automatic proposal protocol completely, including its JSON examples.
    system = system.replace('控制器负责保存、静态校验、执行、复查和预算；本次响应后控制器会推进一步。',
        '你自主决定查询、建草稿、修改、执行、检查、比较和提交的顺序；程序负责工具校验、预算和持久化。')
    start = system.index('需要补充信息时可一次批量请求只读工具')
    end = system.index('仅确实缺少用户必须提供', start)
    system = system[:start] + system[end:]
    start = system.index('只读工具参数：')
    end = system.index('本次职责：', start)
    system = system[:start] + system[end:]
    system = system.replace('本次职责：根据图片和当前状态提出一个完整可执行的Pipeline，或对当前草稿做局部修改。',
        '本次职责：根据最新证据决定下一次工具调用。保存或执行之后都会返回给你，不会自动提交复查。')
    system = system.replace('首次propose同时给出understanding', '首次create_draft的arguments同时给出understanding')
    system = system.replace('可在propose或edit动作顶层返回contract_updates和memory_updates',
                            '可在create_draft或edit_draft的arguments返回contract_updates和memory_updates')
    catalog = system.split('可用目录：', 1)[1]
    system = system.split('新方案动作：', 1)[0]
    messages[0]['content'] = system + (
        '只返回一个JSON：{"kind":"tool","tool":"工具名","arguments":{工具参数}}。'
        'create_draft只保存校验，不自动执行；必须显式execute_pipeline(draft_id,revision)。'
        '执行返回实际experiment_id。可以继续检查、修改、运行不同方案、compare_candidates；'
        '准备好交付时显式submit_experiment(experiment_id,reason)，随后才由独立节点验收。'
        '可提交本轮较早的成功实验；不能提交失败或已被独立验收拒绝的实验。'
        '没有执行额度时仍可检查、比较和提交已有结果。调用与执行共用canonical_state.budget，'
        '每次决策消耗一次模型调用；为提交后的独立验收留出调用和时间。'
        '最新工具返回位于canonical_state.last_tool_result；草稿目录位于draft_catalog。'
        'create_draft首次understanding包括task_summary、target_defect、normal_context、'
        'output_requirements、acceptance_criteria、target_constraints、rendering。'
        '只有确实缺少用户必要信息才返回{"kind":"needs_input","reason":"原因"}。'
        '工具参数：' + json.dumps(tool_schemas(), ensure_ascii=False) +
        '可用目录：' + catalog)
    return messages
