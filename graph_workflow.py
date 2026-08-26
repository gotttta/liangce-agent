"""Backward-compatible entry point for the canonical Agent graph.

New code should import ``run_agent_graph`` from ``core.agent_graph``. This
module remains so existing scripts and regression tests keep working.
"""

from core.agent_graph import build_agent_graph, run_agent_graph
from providers.vision import build_runtime_provider


def build_graph(provider=None, max_iterations=2):
    """Compatibility alias for callers that used the historical graph API."""
    return build_agent_graph(
        provider=provider or build_runtime_provider(),
        max_candidates=3,
        checkpointer=False,
    )


def run_graph(
    target_image_path,
    description,
    output_root="outputs",
    reference_annotation_path=None,
    reference_examples=None,
    feedback_brush_path=None,
    feedback_text=None,
    provider=None,
    unit="pixel",
    max_iterations=2,
    existing_run_dir=None,
    initial_iteration=0,
):
    return run_agent_graph(
        target_image_path=target_image_path,
        description=description,
        output_root=existing_run_dir or output_root,
        reference_annotation_path=reference_annotation_path,
        reference_examples=reference_examples,
        unit=unit,
        max_candidates=3,
        provider=provider or build_runtime_provider(),
        previous_state={
            "iteration": initial_iteration - 1,
            "feedback_brush_path": feedback_brush_path,
            "feedback_text": feedback_text,
        } if initial_iteration else None,
    )
