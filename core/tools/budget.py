"""One request-scoped budget, enforced at dispatch and execution boundaries."""
from dataclasses import dataclass, field
from core.tools.contracts import TOOL_SPECS, ToolError


@dataclass
class ToolBudget:
    limits: dict = field(default_factory=lambda: {
        "discovery": 3, "inspection": 8, "navigation": 4, "experiment": 6, "execution": 2, "comparison": 2,
        "editing": 8, "validation": 6, "task": 2, "submission": 3})
    used: dict = field(default_factory=dict)

    def remaining(self, category):
        return max(0, self.limits[category] - self.used.get(category, 0))

    def consume(self, category):
        if not self.remaining(category):
            raise ToolError("budget_exhausted", f"{category} budget exhausted")
        self.used[category] = self.used.get(category, 0) + 1

    def available(self):
        return [name for name, spec in TOOL_SPECS.items()
                if (self.remaining(spec.budget) or (name == "inspect_experiment" and self.remaining("navigation")))
                and (name != "execute_pipeline" or self.remaining("execution"))
                and (name != "compare_candidates" or self.remaining("comparison"))]

    def snapshot(self):
        return {key: self.remaining(key) for key in self.limits}
