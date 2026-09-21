"""Immutable requirements for automatic revisions; user changes carry provenance."""
from copy import deepcopy

FIELDS = ('task_summary', 'target_defect', 'normal_context', 'output_requirements',
          'acceptance_criteria', 'rendering', 'target_constraints')


def establish_contract(state, understanding):
    from providers.vision import normalize_acceptance_criteria, _extract_explicit_count
    existing = deepcopy(state.get('task_contract') or {})
    message = state.get('description') or ''
    if existing:
        # A source-backed explicit user edit can update a named requirement.
        changed = []
        for update in understanding.get('contract_updates') or []:
            quote, field = update.get('source_quote'), update.get('field')
            if (field not in FIELDS or not isinstance(quote, str) or not quote.strip()
                    or quote not in message or message == existing.get('source_text')):
                continue
            if not any(word in quote.lower() for word in ('改为', '改成', '更改', '不要', '只要', 'change', 'instead')):
                continue
            value = update.get('value')
            expected_type = dict if field in {'acceptance_criteria', 'rendering', 'target_constraints'} else list if field == 'output_requirements' else str
            if not isinstance(value, expected_type):
                continue
            existing[field] = deepcopy(value)
            changed.append({'field': field, 'source_quote': quote})
        if changed:
            existing['acceptance_criteria'] = normalize_acceptance_criteria(existing.get('acceptance_criteria'), task_summary=existing.get('task_summary') or message, output_requirements=existing.get('output_requirements'))
            existing['version'] = existing.get('version', 1) + 1
            existing['changes'] = changed
            existing['source_text'] = message
        return existing
    contract = {key: deepcopy(understanding.get(key)) for key in FIELDS}
    contract.update(version=1, source='user_request_initial_interpretation', source_text=message,
                    input_sha256=state.get('input_sha256'), unit=state.get('unit', 'pixel'))
    contract['task_summary'] = contract['task_summary'] or message
    criteria = normalize_acceptance_criteria(contract.get('acceptance_criteria'),
        task_summary=contract['task_summary'], output_requirements=contract.get('output_requirements'))
    constraints = contract.get('target_constraints') or {}
    count = _extract_explicit_count(message)
    if count is not None:
        constraints = {**constraints, 'expected_count': count, 'count_source': 'user_explicit'}
        criteria.update(count_policy='exact', expected_count=count, count_source='user_explicit')
    if count is None:
        constraints.pop('expected_count', None)
        if criteria.get('count_policy') == 'exact':
            criteria.pop('expected_count', None)
            criteria['count_policy'] = 'unspecified'
    contract['acceptance_criteria'] = criteria
    contract['target_constraints'] = constraints
    return contract


def apply_contract(understanding, contract):
    return {**understanding, **{key: deepcopy(contract[key]) for key in FIELDS if contract.get(key) is not None}}


def check_delivery(candidate, contract):
    """Deterministic delivery/count checks; visual judgement remains separate."""
    quality = candidate.get('quality') or {}
    measurements = candidate.get('measurements') or {}
    criteria = contract.get('acceptance_criteria') or {}
    constraints = contract.get('target_constraints') or {}
    issues = []
    summary = measurements.get('summary') or {}
    count = summary.get('count')
    expected = criteria.get('expected_count', constraints.get('expected_count'))
    if (criteria.get('count_source', constraints.get('count_source')) == 'user_explicit'
            and expected is not None and count != expected):
        issues.append('component_count_mismatch')
    if constraints.get('allow_empty') is False and count == 0:
        issues.append('empty_result_forbidden')
    if constraints.get('max_coverage') is not None and quality.get('coverage', 0) > float(constraints['max_coverage']):
        issues.append('coverage_exceeds_task_limit')
    if constraints.get('max_components') is not None and count is not None and count > int(constraints['max_components']):
        issues.append('component_count_exceeds_task_limit')
    outputs = measurements.get('structured_outputs') or {}
    available = set(outputs)
    if quality.get('output_kind') != 'structured' and 'coverage' in quality:
        available.update(('mask', 'contours', 'bbox', 'measurements'))
    if quality.get('output_kind') == 'structured':
        for name, item in outputs.items():
            if item.get('kind') == 'image':
                available.update(('image', 'result_image', 'final_image'))
            if item.get('kind') == 'metadata':
                available.add('measurements')
                available.update((item.get('data') or {}).keys())
                available.update(item.get('fields') or [])
    invalidated = set(quality.get('invalidated_outputs') or [])
    import re
    declared = [*(contract.get('output_requirements') or []), *(criteria.get('requested_output') or [])]
    # Artifact identifiers have an exact machine contract. Prose belongs to the
    # visual/task review and must not be interpreted as a literal filename.
    identifiers = {name for name in declared if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name)}
    for name in sorted(identifiers):
        if name not in available or name in invalidated:
            issues.append('missing_output:' + name)
    return {'passed': not issues, 'issues': issues,
            'semantic_requirements_for_review': [name for name in declared if name not in identifiers]}
