from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    function: object
    input_type: type
    output_type: type


class OperatorRegistry:
    def __init__(self):
        self._operators = {}

    def register(self, name, function, input_type, output_type):
        if not name or not isinstance(name, str):
            raise ValueError("operator name must be a non-empty string")
        if name in self._operators:
            raise ValueError(f"operator already registered: {name}")
        self._operators[name] = OperatorDefinition(name, function, input_type, output_type)

    def names(self):
        return tuple(sorted(self._operators))

    def definition(self, name):
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"unknown operator: {name}") from exc

    def run(self, name, artifact, **params):
        definition = self.definition(name)
        if not isinstance(artifact, definition.input_type):
            raise TypeError(
                f"operator {name} expects {definition.input_type.__name__}, "
                f"got {type(artifact).__name__}"
            )
        result = definition.function(artifact, **params)
        if not isinstance(result.artifact, definition.output_type):
            raise TypeError(
                f"operator {name} returned {type(result.artifact).__name__}, "
                f"expected {definition.output_type.__name__}"
            )
        return result
