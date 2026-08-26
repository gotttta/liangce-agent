from core.operators.image import register_image_operators
from core.operators.mask import register_mask_operators
from core.operators.defect import register_defect_operators
from core.operators.registry import OperatorRegistry
from core.operators.generated import (
    GeneratedOperatorSpec,
    normalize_generated_operator,
    normalize_generated_operators,
    register_generated_operator,
    validate_generated_source,
)
from core.operators.types import (
    ContourArtifact,
    ImageArtifact,
    MaskArtifact,
    MetadataArtifact,
    OperatorResult,
)


def build_default_registry(generated_operators=None):
    registry = OperatorRegistry()
    register_image_operators(registry)
    register_mask_operators(registry)
    register_defect_operators(registry)
    for spec in generated_operators or ():
        register_generated_operator(registry, spec)
    return registry


__all__ = [
    "ContourArtifact",
    "ImageArtifact",
    "MaskArtifact",
    "MetadataArtifact",
    "OperatorRegistry",
    "OperatorResult",
    "build_default_registry",
    "GeneratedOperatorSpec",
    "normalize_generated_operator",
    "normalize_generated_operators",
    "register_generated_operator",
    "validate_generated_source",
]
