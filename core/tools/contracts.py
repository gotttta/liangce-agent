"""Provider-independent Agent Tool schemas and machine-readable results."""
from dataclasses import dataclass, field


class ToolError(ValueError):
    def __init__(self, code, message, *, retryable=False, details=None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details

    def as_dict(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable,
                **({'details': self.details} if self.details else {})}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict
    required: tuple
    budget: str
    effect: str = "read"
    version: str = "1.0.0"

    @property
    def parameters(self):
        return {"type": "object", "properties": self.properties,
                "required": list(self.required), "additionalProperties": False}

    def function_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def validate(self, arguments):
        _validate(arguments, self.parameters, "arguments")


def _validate(value, schema, path):
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str),
             "integer": isinstance(value, int) and not isinstance(value, bool)}
    if not valid.get(kind, True):
        raise ToolError("invalid_arguments", f"{path} must be {kind}", retryable=True)
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError("invalid_arguments", f"{path}: unsupported value", retryable=True)
    if kind == "integer" and not schema.get("minimum", float("-inf")) <= value <= schema.get("maximum", float("inf")):
        raise ToolError("invalid_arguments", f"{path}: out of range", retryable=True)
    if kind == "object" and "properties" in schema:
        missing = set(schema.get("required", [])) - value.keys()
        extra = value.keys() - schema["properties"].keys()
        if missing or (extra and schema.get("additionalProperties") is False):
            raise ToolError("invalid_arguments", f"{path}: missing={sorted(missing)}, unknown={sorted(extra)}", retryable=True)
        for key, item in value.items():
            _validate(item, schema["properties"].get(key, {}), f"{path}.{key}")
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            raise ToolError("invalid_arguments", f"{path}: invalid item count", retryable=True)
        for item in value:
            _validate(item, schema.get("items", {}), path + "[]")


def strings(minimum, maximum):
    return {"type": "array", "items": {"type": "string"}, "minItems": minimum, "maxItems": maximum}


TOOL_SPECS = {spec.name: spec for spec in [
    ToolSpec('save_task', '先保存任务理解和验收标准，不含算法。保存后当前会话的验收要求固定。',
             {'understanding': {'type': 'object'}}, ('understanding',), 'task', 'write'),
    ToolSpec('create_draft', '保存一个算法草稿并自动静态检查。语法失败也会保存，返回精确诊断；不消耗执行次数。',
             {'pipeline': {'type': 'object'}, 'change_reason': {'type': 'string'},
              'expected_change': {'type': 'string'}, 'parent_experiment_id': {'type': 'string'}},
             ('pipeline', 'change_reason'), 'editing', 'write'),
    ToolSpec('read_draft', '读取当前草稿版本。path为JSON Pointer；空串读取完整pipeline，例如/generated_operators/0/source读取源码。',
             {'draft_id': {'type': 'string'}, 'path': {'type': 'string'}}, ('draft_id',), 'inspection'),
    ToolSpec('edit_draft', '按当前版本原子应用局部修改并自动检查。path为已存在的JSON Pointer；字符串old必须恰好匹配一次并替换为new；非字符串需与old完全相等再替换。冲突时read_draft，不猜测覆盖。',
             {'draft_id': {'type': 'string'}, 'base_revision': {'type': 'integer', 'minimum': 1},
              'change_reason': {'type': 'string'}, 'expected_change': {'type': 'string'},
              'edits': {'type': 'array', 'minItems': 1, 'maxItems': 20, 'items': {
                  'type': 'object', 'properties': {'path': {'type': 'string'}, 'old': {}, 'new': {}},
                  'required': ['path', 'old', 'new'], 'additionalProperties': False}}},
             ('draft_id', 'base_revision', 'edits', 'change_reason'), 'editing', 'write'),
    ToolSpec('submit_experiment', '以ID提交已执行的当前任务实验供正式工作流独立复查；不重写源码，不代表验收通过。',
             {'experiment_id': {'type': 'string'}, 'reason': {'type': 'string'}},
             ('experiment_id', 'reason'), 'submission', 'write'),
    ToolSpec("query_operators", "查询算子的参数与输入输出定义。", {"names": strings(1, 30)}, ("names",), "discovery"),
    ToolSpec("load_skill", "加载 Skill 的 SKILL.md 正文；六个内置 Skill 的正文包含核心量测口径与验收项。仅需延伸资料时指定相对 resource 路径读取；脚本只返回源码，不执行。",
             {"name": {"type": "string"}, "version": {"type": "string"}, "resource": {"type": "string"}}, ("name",), "discovery"),
    ToolSpec("inspect_artifact", "查看当前会话实验的中间产物。", {"ids": strings(1, 3)}, ("ids",), "inspection"),
    ToolSpec("inspect_experiment", "按实验ID读取完整报告的精确分页，或查看原图/叠加图/最终Mask的对齐原分辨率局部图。摘要缺省不是不存在；检查遗漏需结合全图。region与报告参数互斥。",
             {"experiment_id": {"type": "string"},
              "report": {"type": "string", "enum": ["measurements", "outputs", "quality_report", "operator_trace", "pipeline"]},
              "selector": {"type": "string", "description": "JSON Pointer，空串为根；如 /results、/measurements/data/components。"},
              "offset": {"type": "integer", "minimum": 0},
              "limit": {"type": "integer", "minimum": 1, "maximum": 20},
              "region": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 4, "maxItems": 4,
                         "description": "原图像素[left,top,right,bottom]，右下不包含，最大1024×1024，不缩放。"}},
             ("experiment_id",), "inspection"),
    ToolSpec("execute_pipeline", "执行一次实验；先检查结果再决定下一次修改。执行成功不表示验收通过。",
             {"pipeline": {"type": "object"}, 'draft_id': {'type': 'string'},
              'revision': {'type': 'integer', 'minimum': 1}, "hypothesis": {"type": "string"},
              "change_reason": {"type": "string"}, "expected_change": {"type": "string"},
              "parent_experiment_id": {"type": "string"}}, (), "validation", "experiment_write"),
    ToolSpec("compare_candidates", "按需比较相同输入和约束的已有实验。仅在方法有依据需比较、修改停滞或防止退步时使用；reason说明原因。差异不是准确率。",
             {"experiment_ids": strings(2, 3), "reason": {"type": "string"}}, ("experiment_ids",), "experiment", "experiment_write"),
]}


@dataclass
class ToolResult:
    call_id: str
    data: dict = field(default_factory=dict)
    error: ToolError | None = None
    images: list = field(default_factory=list)

    def as_dict(self):
        return {"call_id": self.call_id, "status": "error" if self.error else "success",
                "data": self.data, "error": self.error.as_dict() if self.error else None}
