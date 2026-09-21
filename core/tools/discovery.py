"""Read-only, bounded disclosure of operators, skills and experiment images."""
from core.pipelines.dsl import pipeline_operator_catalog
from core.skills import SkillRegistry


def operator_index():
    return [{key: item[key] for key in ('name', 'description')}
            for item in pipeline_operator_catalog()]


def generated_catalog(context=None):
    from core.operator_library import OperatorLibrary
    from pathlib import Path
    from hashlib import sha256
    operators = OperatorLibrary(Path(__file__).resolve().parents[2] / 'workspace/operators').list_operators()
    operators += ((context or {}).get('previous_pipeline') or {}).get('generated_operators', [])
    result = {}
    for item in operators:
        digest = sha256(item.get('source', '').encode()).hexdigest()
        spec = {**item, 'source_sha256': digest, 'operator_id': item['name'] + '@' + digest}
        result[spec['operator_id']] = spec
    return result


def query_operators(names, context=None):
    if not isinstance(names, list) or not names or len(names) > 30 or not all(isinstance(n, str) for n in names):
        raise ValueError('names must contain 1 to 30 operator names')
    catalog = {item['name']: item for item in pipeline_operator_catalog()}
    generated = generated_catalog(context)
    catalog.update(generated)
    for item in reversed(list(generated.values())):
        catalog.setdefault(item['name'], item)
    unknown = set(names) - catalog.keys()
    if unknown:
        raise ValueError(f'unknown operators: {sorted(unknown)}')
    return [catalog[name] for name in dict.fromkeys(names)]


def available_artifacts(context):
    attempts = ((context or {}).get('execution_feedback') or {}).get('attempts') or []
    return {item['id']: item for attempt in attempts for item in attempt.get('artifacts', [])}


def dispatch_discovery(action, context, skill_root, registry=None):
    tool, args = action.get('tool'), action.get('arguments', {})
    if not isinstance(args, dict):
        raise ValueError('arguments must be an object')
    if tool == 'query_operators':
        return {'operators': query_operators(args.get('names'), context)}, []
    if tool == 'load_skill':
        skills = registry or SkillRegistry(skill_root)
        if 'resource' in args:
            return {'resource': skills.read_resource(args.get('name'), args['resource'], version=args.get('version'))}, []
        skill = skills.get(args.get('name'), version=args.get('version'))
        if skill is None:
            raise ValueError('unknown skill')
        if 'pipeline_template' not in skill:
            return {'skill': skill}, []
        # Compatibility for accepted legacy workspace JSON templates only.
        template = skill['pipeline_template']
        names = [node['operator'] for node in template.get('nodes', [])]
        return {'skill': skill, 'operators': query_operators(names)}, []
    if tool == 'inspect_artifact':
        ids = args.get('ids')
        if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or not all(isinstance(i, str) for i in ids):
            raise ValueError('ids must contain 1 to 3 artifact IDs')
        allowed = available_artifacts(context)
        if any(key not in allowed for key in ids):
            raise ValueError('artifact is not part of this revision context')
        selected = [allowed[key] for key in ids]
        return {'artifacts': selected}, [item['preview_path'] for item in selected]
    raise ValueError(f'unknown discovery tool: {tool}')
