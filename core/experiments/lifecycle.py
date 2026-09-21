"""Experiment review history and explicitly verified rollback baselines."""
from copy import deepcopy
from pathlib import Path

from core.experiments.artifacts import experiment_scope, update_experiment


def compatible_baseline(state):
    baseline = state.get('verified_baseline') or {}
    if (baseline.get('acceptance_status') not in {'passed', 'accepted'}
            or baseline.get('scope') != experiment_scope(state['target_image_path'], context=state)
            or not (Path(baseline.get('directory', '')) / 'result_annotation.png').is_file()):
        return None
    return baseline


def record_review(state):
    """Execution, objective checks and visual acceptance remain separate facts."""
    review = state.get('review') or {}
    selected_id = state.get('selected_experiment_id')
    attempts = deepcopy(state.get('candidate_attempts') or [])
    baseline = compatible_baseline(state)
    for attempt in attempts:
        selected = attempt.get('experiment_id') == selected_id
        status = 'pending'
        if attempt.get('status') != 'selected_for_review':
            status = 'rejected'
        elif selected:
            if (review.get('acceptance') or {}).get('overall_passed'):
                status = 'passed'
            elif review.get('decision') in {'revise', 'failed'}:
                status = 'rejected'
        changes = {'acceptance_status': status}
        if selected or attempt.get('status') != 'selected_for_review':
            changes['review'] = deepcopy(review)
        update_experiment(attempt, **changes)
        if status == 'passed' and attempt.get('experiment_id') and attempt.get('directory'):
            baseline = deepcopy(attempt)
    records = {item['experiment_id']: item for item in state.get('experiment_records', []) if item.get('experiment_id')}
    records.update({item['experiment_id']: item for item in attempts if item.get('experiment_id')})
    history = deepcopy(state.get('experiment_history') or [])
    if history:
        history[-1].update(candidate_attempts=attempts, review=deepcopy(review),
                           selected_experiment_id=selected_id)
    result = {**state, 'candidate_attempts': attempts, 'experiment_records': list(records.values()),
              'experiment_history': history, 'verified_baseline': baseline}
    return result


def record_human_review(state, response):
    """Human decisions apply to the displayed version, including rollback results."""
    records = deepcopy(state.get('experiment_records') or [])
    selected_id = state.get('selected_experiment_id')
    selected = next((item for item in records if item.get('experiment_id') == selected_id), None)
    if selected is None and state.get('verified_baseline') and state['verified_baseline'].get('experiment_id') == selected_id:
        selected = deepcopy(state['verified_baseline'])
        records.append(selected)
    if selected is None or selected.get('status') != 'selected_for_review':
        return state
    accepted = response.get('action') == 'accept'
    update_experiment(selected, acceptance_status='accepted' if accepted else 'rejected',
                      human_review=deepcopy(response))
    attempts = [deepcopy(selected) if item.get('experiment_id') == selected_id else item
                for item in state.get('candidate_attempts') or []]
    history = deepcopy(state.get('experiment_history') or [])
    for experiment in history:
        experiment['candidate_attempts'] = [deepcopy(selected) if item.get('experiment_id') == selected_id else item
                                           for item in experiment.get('candidate_attempts', [])]
    baseline = next((item for item in reversed(records)
                     if compatible_baseline({**state, 'verified_baseline': item})), None)
    return {**state, 'verified_baseline': deepcopy(baseline), 'experiment_records': records,
            'candidate_attempts': attempts, 'experiment_history': history}


def revision_evidence(state):
    """Expose scoped history, including unsuccessful trials, for diagnosis."""
    scope = experiment_scope(state['target_image_path'], context=state)
    records = [item for item in state.get('experiment_records', []) if item.get('scope') == scope]
    baseline = compatible_baseline(state)
    if baseline and not any(item.get('experiment_id') == baseline['experiment_id'] for item in records):
        records.append(baseline)
    return records
