from dataclasses import dataclass, replace
import inspect


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    function: object
    input_type: type
    output_type: type
    input_ports: dict
    version: str = "1.0.0"
    description: str = ""
    model_visible: bool = False
    legacy_allowed: bool = True


class OperatorRegistry:
    def __init__(self):
        self._operators = {}

    def register(
        self,
        name,
        function,
        input_type,
        output_type,
        *,
        input_ports=None,
        version="1.0.0",
        description="",
        model_visible=False,
        legacy_allowed=True,
    ):
        if not name or not isinstance(name, str):
            raise ValueError("operator name must be a non-empty string")
        if name in self._operators:
            raise ValueError(f"operator already registered: {name}")
        if input_ports is None:
            parameters = tuple(inspect.signature(function).parameters)
            input_ports = {parameters[0] if parameters else "input": input_type}
        ports = dict(input_ports)
        if not ports or not all(isinstance(port, str) and port for port in ports):
            raise ValueError("operator input_ports must contain named ports")
        if input_type not in ports.values():
            raise ValueError("operator input_type must be represented by input_ports")
        self._operators[name] = OperatorDefinition(
            name=name,
            function=function,
            input_type=input_type,
            output_type=output_type,
            input_ports=ports,
            version=str(version),
            description=str(description),
            model_visible=bool(model_visible),
            legacy_allowed=bool(legacy_allowed),
        )

    def configure(self, name, **metadata):
        """Apply catalog metadata after legacy operator registration."""
        definition = self.definition(name)
        allowed = {"version", "description", "model_visible", "legacy_allowed"}
        unknown = set(metadata) - allowed
        if unknown:
            raise ValueError(f"unknown operator metadata: {sorted(unknown)}")
        self._operators[name] = replace(definition, **metadata)

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

    def run_inputs(self, name, artifacts, **params):
        """Run a v3 operator with typed, named artifact inputs."""
        definition = self.definition(name)
        if not isinstance(artifacts, dict):
            raise TypeError("operator artifacts must be a mapping")
        expected_ports = set(definition.input_ports)
        actual_ports = set(artifacts)
        if actual_ports != expected_ports:
            raise ValueError(
                f"operator {name} expects input ports {sorted(expected_ports)}, "
                f"got {sorted(actual_ports)}"
            )
        for port, expected_type in definition.input_ports.items():
            artifact = artifacts[port]
            if not isinstance(artifact, expected_type):
                raise TypeError(
                    f"operator {name} input {port} expects {expected_type.__name__}, "
                    f"got {type(artifact).__name__}"
                )
        result = definition.function(**artifacts, **params)
        if not isinstance(result.artifact, definition.output_type):
            raise TypeError(
                f"operator {name} returned {type(result.artifact).__name__}, "
                f"expected {definition.output_type.__name__}"
            )
        return result
