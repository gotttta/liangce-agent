import base64
import json
import os
import re
from pathlib import Path

from agent_types import normalize_strategy
from core.agent_events import emit_llm_chunk, emit_llm_request, emit_llm_response, emit_thinking


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_VISION_MODEL = "qwen3.7-plus"
NODE_TIMEOUT_SECONDS = 90
NODE_MAX_RETRIES = 3


def load_env_file(path=None):
    env_path = Path(path or ROOT / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _stream_chunk_text(chunk):
    """Read text from dict-like or SDK object streaming chunks."""
    choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
    if delta is None:
        return ""
    content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(getattr(item, "text", ""))
            for item in content
        )
    return content or ""


def build_runtime_provider(env_path=None):
    load_env_file(env_path)
    api_key = os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("ALIYUN_BASE_URL")
    if not api_key or not base_url:
        raise ValueError(
            "Missing Alibaba Cloud vision configuration. "
            "Please set DASHSCOPE_API_KEY or ALIYUN_API_KEY, and ALIYUN_BASE_URL in .env."
        )
    return AliyunVisionProvider(api_key=api_key, base_url=base_url)


class MockVisionProvider:
    def understand_task(
        self,
        target_image_path,
        description,
        previous_context=None,
        reference_examples=None,
        progress_callback=None,
    ):
        strategy = self.create_strategy(target_image_path, description)
        polarity = strategy["segmentation"]["method"]
        comparison = "<" if polarity == "dark_threshold" else ">"
        return {
            "task_summary": description,
            "target_defect": "需要由用户结合候选结果确认的视觉异常",
            "normal_context": "从样本图中的多数结构推断",
            "ambiguities": ["缺陷边界和可接受变化尚未通过多样本确认"],
            "questions": [],
            "output_requirements": ["mask", "measurements"],
            "acceptance_criteria": {
                "task_goal": description,
                "requested_output": ["mask", "measurements"],
                "visual_checks": ["标注应覆盖用户描述的目标，并尽量贴合目标边界"],
                "failure_examples": ["明显漏标、误标，或标注边界偏离目标"],
            },
            "candidate_plans": [
                {
                    "name": "local_threshold_baseline",
                    "hypothesis": "缺陷可由局部灰度异常形成初始候选",
                    "operators": ["normalize", strategy["segmentation"]["method"], "morphology", "filter_components"],
                    "new_operator_needed": False,
                }
            ],
            "candidate_pipelines": [{
                "name": "mock_generated_threshold",
                "hypothesis": "Test-only generated threshold stage.",
                "pipeline": {
                    "name": "mock_generated_threshold",
                    "steps": [{
                        "id": "final_mask",
                        "op": "mock_threshold_mask",
                        "input": "image",
                        "params": {},
                    }],
                    "generated_operators": [{
                        "name": "mock_threshold_mask",
                        "input_artifact": "ImageArtifact",
                        "output_artifact": "MaskArtifact",
                        "atomic": True,
                        "description": "Test-only threshold stage",
                        "source": (
                            "def apply(data, params):\n"
                            f"    return data {comparison} np.mean(data)"
                        ),
                    }],
                },
            }],
            "recommended_strategy": strategy,
            "confidence": 0.5,
        }

    def create_strategy(
        self,
        target_image_path,
        description,
        reference_annotation_path=None,
        previous_state=None,
    ):
        text = (description or "").lower()
        method = "dark_threshold" if any(word in text for word in ("暗", "dark", "black")) else "bright_threshold"
        return normalize_strategy({
            "defect_type": "user_defined_defect",
            "measurement_type": "area_count",
            "visual_observation": {
                "defect_appearance": "visual anomaly inferred from user description",
                "background_pattern": "unknown",
                "polarity": "dark_on_bright" if method == "dark_threshold" else "bright_on_dark",
            },
            "segmentation": {
                "method": method,
                "sensitivity": 1.8,
                "min_area_px": 20,
                "morphology": "open_then_close",
            },
            "confidence": 0.5,
            "notes": ["Mock provider used; no remote multimodal model was called."],
        })

    def review_candidates(
        self,
        target_image_path,
        description,
        candidates,
        reference_examples=None,
        acceptance_criteria=None,
    ):
        completed = [item for item in candidates if item.get("status") in {"completed", "selected_for_review"}]
        if not completed:
            return {"decision": "cannot_determine", "selected_candidate": None, "reason": "没有可复查的候选结果"}
        return {
            "decision": "present",
            "selected_candidate": completed[0].get("name"),
            "reason": "Mock provider 选择首个可执行候选；需要用户检查视觉准确性。",
            "observed_issues": [],
        }


class FixedStrategyProvider:
    """Expose an already-approved strategy through the graph provider contract."""

    def __init__(self, strategy):
        self.strategy = normalize_strategy(strategy)

    def create_strategy(
        self,
        target_image_path,
        description,
        reference_annotation_path=None,
        previous_state=None,
    ):
        return self.strategy


class AliyunVisionProvider:
    def __init__(
        self,
        api_key=None,
        base_url=None,
        model=None,
        timeout_seconds=None,
        max_retries=None,
    ):
        self.api_key = api_key or os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("ALIYUN_BASE_URL", DEFAULT_ALIYUN_BASE_URL)
        self.model = model or os.getenv("ALIYUN_VISION_MODEL", DEFAULT_ALIYUN_VISION_MODEL)
        self.timeout_seconds = int(timeout_seconds or NODE_TIMEOUT_SECONDS)
        self.max_retries = int(NODE_MAX_RETRIES if max_retries is None else max_retries)
        if not self.api_key:
            raise ValueError("Missing ALIYUN_API_KEY or DASHSCOPE_API_KEY")

    def understand_task(
        self,
        target_image_path,
        description,
        previous_context=None,
        reference_examples=None,
        progress_callback=None,
    ):
        from openai import OpenAI

        emit_thinking("正在调用视觉模型理解任务...", "understand_task")
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        messages = build_task_understanding_messages(
            target_image_path,
            description,
            previous_context=previous_context,
            reference_examples=reference_examples,
        )
        emit_llm_request("Aliyun", self.model, len(messages), has_images=True)
        validation_error = None
        for _ in range(2):
            if validation_error:
                emit_thinking(f"代码校验失败，重新生成: {validation_error}", "validate_code")
                messages = [*messages, {
                    "role": "user",
                    "content": (
                        "上一版候选代码未通过本地安全校验："
                        f"{validation_error}。请重新输出完整JSON；只使用允许的np函数、"
                        "params.get，以及数组的astype/copy方法和shape/ndim/size属性。"
                    ),
                }]
            content = self._complete_streaming(client, messages, progress_callback=progress_callback)
            emit_llm_response("Aliyun", content[:500])
            try:
                result = normalize_task_understanding(
                    extract_json_object(content),
                    task_description=(previous_context or {}).get("original_task_goal") or description,
                )
                emit_thinking("任务理解完成，生成了候选算法", "understand_complete")
                return result
            except ValueError as exc:
                validation_error = str(exc)
        raise ValueError(f"Vision provider could not produce a valid candidate pipeline: {validation_error}")

    def create_strategy(
        self,
        target_image_path,
        description,
        reference_annotation_path=None,
        previous_state=None,
    ):
        understanding = self.understand_task(
            target_image_path,
            description,
            previous_context=previous_state,
        )
        return normalize_strategy(understanding.get("recommended_strategy"))

    def review_candidates(
        self,
        target_image_path,
        description,
        candidates,
        reference_examples=None,
        acceptance_criteria=None,
        progress_callback=None,
    ):
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        review_messages = build_candidate_review_messages(
            target_image_path,
            description,
            candidates,
            reference_examples=reference_examples,
            acceptance_criteria=acceptance_criteria,
        )
        emit_llm_request("Aliyun", self.model, len(review_messages), has_images=True)
        review_content = self._complete_streaming(
            client,
            review_messages,
            progress_callback=progress_callback,
        )
        emit_llm_response("Aliyun", review_content[:500])
        names = {str(item.get("name")) for item in candidates if item.get("status") in {"completed", "selected_for_review"}}
        return normalize_candidate_review(extract_json_object(review_content), names)

    def _complete_streaming(self, client, messages, progress_callback=None):
        """Collect an OpenAI-compatible streamed response and expose chunks."""
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            response = client.chat.completions.create(**request_kwargs)
        except TypeError:
            # Some test doubles and older SDKs do not accept ``stream``.
            request_kwargs.pop("stream")
            response = client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or ""

        chunks = []
        try:
            iterator = iter(response)
        except TypeError:
            return response.choices[0].message.content or ""
        for chunk in iterator:
            text = _stream_chunk_text(chunk)
            if not text:
                continue
            chunks.append(text)
            emit_llm_chunk(text, provider="Aliyun", model=self.model)
            if progress_callback:
                progress_callback({
                    "type": "llm_chunk",
                    "provider": "Aliyun",
                    "model": self.model,
                    "content": text,
                })
        return "".join(chunks)


def build_task_understanding_messages(
    target_image_path,
    description,
    previous_context=None,
    reference_examples=None,
):
    from core.pipelines.dsl import pipeline_operator_catalog
    from core.operator_library import OperatorLibrary

    # Built-ins are tools the model may compose; execution remains validated
    # by the DSL and sandbox, so visibility does not grant arbitrary code access.
    operator_catalog = pipeline_operator_catalog(include_builtin=True)
    reusable_operators = OperatorLibrary(ROOT / "workspace" / "operators").list_operators()
    schema = {
        "task_summary": "string",
        "target_defect": "string",
        "normal_context": "string",
        "ambiguities": ["string"],
        "questions": ["string"],
        "output_requirements": ["mask|contours|bbox|points|measurements"],
        "acceptance_criteria": {
            "task_goal": "用一句话说明结果必须完成什么",
            "requested_output": ["用户要求的输出类型"],
            "visual_checks": ["可直接通过原图和结果图核对的任务专属条件"],
            "failure_examples": ["哪些可见现象表示结果不合格"],
        },
        "candidate_plans": [
            {
                "name": "string",
                "hypothesis": "string",
                "operators": ["generic operator names"],
                "new_operator_needed": False,
            }
        ],
        "candidate_pipelines": [
            {
                "name": "string",
                "hypothesis": "string",
                "pipeline": {
                    "name": "string",
                    "steps": [{
                        "id": "final_mask",
                        "op": "custom_operator_name",
                        "input": "image",
                        "params": {},
                    }],
                    "generated_operators": [
                        {
                            "name": "custom_operator_name",
                            "input_artifact": "ImageArtifact",
                            "output_artifact": "MaskArtifact",
                            "atomic": True,
                            "description": "why the catalog is insufficient",
                            "source": "def apply(data, params):\\n    return np.zeros_like(data, dtype=np.bool_)",
                        }
                    ],
                },
            }
        ],
        "target_constraints": {
            "expected_shape": "elongated|elongated_ellipse|compact|unknown",
            "max_coverage": 0.35,
            "expected_count": None,
            "count_source": "user_explicit|model_observed|unknown",
            "observed_count": None,
        },
        "rendering": {
            "annotation_mode": "contour|mask|bbox",
            "contour_color": "#39FF14",
            "contour_thickness": 2,
            "mask_alpha": 72,
        },
        "recommended_strategy": {
            "defect_type": "user_defined_defect",
            "measurement_type": "area_count",
            "visual_observation": {
                "defect_appearance": "string",
                "background_pattern": "string",
                "polarity": "bright_on_dark|dark_on_bright|mixed|unknown",
            },
            "segmentation": {
                "method": "bright_threshold|dark_threshold|auto_bright_dark_threshold",
                "sensitivity": 1.8,
                "min_area_px": 20,
                "morphology": "none|open|close|open_then_close|close_then_open",
            },
            "confidence": 0.5,
            "notes": ["string"],
        },
        "confidence": 0.5,
    }
    text = (
        "你是通用工业视觉算法开发 Agent 的任务理解节点。"
        "观察用户图片和描述，识别目标缺陷、正常上下文和不确定点，并直接生成一个可执行的CV Pipeline。"
        "如果提供了甲方Handbook标注示例图，它们是不同图片上的few-shot视觉参照："
        "只能学习哪些对象应被标注、边界位置和标注风格；不得复制其像素坐标，"
        "参考图不对应当前图片的像素坐标，也不得据此计算准确率。"
        "默认用户没有计算机视觉或半导体知识。你必须自行选择检测条件、算子和参数，"
        "不得要求用户在执行前确认长宽比、阈值、空洞填充、形态学处理或任何技术方案。"
        "生成最终标注图后，系统才会请用户直观确认标注效果。"
        "questions必须返回空数组；ambiguities只供Agent内部自动决策，不能写成需要用户回答的问题。"
        "必须根据当前图片和用户描述生成acceptance_criteria。visual_checks只能描述最终结果中可直接观察的目标、"
        "覆盖范围、边界或输出形式，不能写阈值、算子、参数等实现方法；failure_examples要说明当前任务中的"
        "明显漏标、误标、边界错误或输出形式错误，不能套用固定缺陷规则。"
        "如果用户没有明确指定数量，不要在验收条件中写‘恰好N个’，应描述为覆盖所有当前可见目标；"
        "不要直接生成mask，不要假设固定缺陷类型。优先复用通用算子；算子库没有且无法组合复用时，"
        "必须生成generated_operators中的自定义算子。每个自定义算子必须是一个原子环节，atomic必须为true；"
        "二值化、轮廓提取、填洞、过滤等应分别是不同的pipeline step，禁止把完整检测算法塞进一个算子。"
        "自定义算子只能实现apply(data, params)，"
        "只能使用np和params.get，不能import、访问文件、网络、系统或执行任意代码；它会在沙箱中运行。"
        "复用本地自定义算子时，必须把工具目录中的完整算子定义原样放入pipeline.generated_operators，"
        "不要重新生成同名源码。"
        "recommended_strategy只是视觉理解摘要和检索特征；实际执行必须来自你生成的candidate_pipelines中的第一个Pipeline。"
        "生成2到3个语义不同的candidate_pipelines条目；它们只能使用下面目录中的已批准算子，或在其generated_operators中完整定义的新算子，"
        "Pipeline必须产生final_mask；contours只有在任务需要轮廓时才加入。"
        "Pipeline Tool Catalog："
        + json.dumps(operator_catalog, ensure_ascii=False)
        + "受限内建Pipeline（无需模型拼接多输入算子）："
        + json.dumps([{
            "name": "periodic_particle_builtin",
            "kind": "builtin_pipeline",
            "description": "已验证的周期背景建模、残差阈值、排除边界和组件过滤流程",
            "params": {"percentile": 97.0, "min_area": 20, "max_components": 3, "roi": None},
        }], ensure_ascii=False)
        + "本地可复用自定义原子算子："
        + json.dumps([
            {
                "name": item.get("name"),
                "input_artifact": item.get("input_artifact"),
                "output_artifact": item.get("output_artifact"),
                "description": item.get("description"),
                "source": item.get("source"),
                "atomic": item.get("atomic", True),
            }
            for item in reusable_operators
        ], ensure_ascii=False)
        + "将检测条件放入target_constraints，将输出类型和颜色、线宽、透明度放入rendering，不要混入分割参数。"
        + "数量约束必须区分来源：只有用户在原始任务中明确说出数量时，才填写expected_count并将count_source设为user_explicit；"
        + "如果只是从图片观察到大约有几个目标，只填写observed_count并将count_source设为model_observed。"
        + "observed_count不是硬性验收条件，禁止把它写入Pipeline的max_components；max_components只能作为与目标数量无关的噪声安全上限。"
        + "只输出JSON，不要Markdown。JSON结构示例："
        + json.dumps(schema, ensure_ascii=False)
        + f"\n用户描述：{description or ''}"
    )
    content = [
        {"type": "text", "text": text},
        {"type": "text", "text": "下面第一张是当前待处理图片。"},
        image_content(target_image_path, "当前待处理图片"),
    ]
    for index, example in enumerate(normalize_reference_examples(reference_examples), start=1):
        content.append({
            "type": "text",
            "text": (
                f"下面是甲方Handbook标注示例 {index}。"
                "它与当前图片没有像素坐标对应关系，只用于学习标注语义和样式。"
                f"示例说明：{example.get('description') or '甲方已标注示例'}"
            ),
        })
        content.append(image_content(example["image_path"], f"Handbook标注示例 {index}"))
    if previous_context:
        original_task_goal = previous_context.get("original_task_goal")
        if original_task_goal:
            content.append({
                "type": "text",
                "text": (
                    "原始任务目标（不可因本轮修正而缩小或替换）："
                    f"{original_task_goal}"
                ),
            })
        execution_feedback = previous_context.get("execution_feedback") or {}
        if execution_feedback.get("status") == "no_usable_annotation":
            content.append({
                "type": "text",
                "text": (
                    "上一轮方法没有生成任何可用标注。请检查attempts中的operator_trace，"
                    "找出候选在哪一步变成空Mask或执行失败。新Pipeline必须实质改变算子顺序、"
                    "处理条件或参数，不能原样重复previous_pipeline。请根据本次执行记录定位原因，"
                    "不要套用某一种目标形状的固定修补规则。"
                ),
            })
        elif execution_feedback.get("status") == "duplicate_pipeline":
            content.append({
                "type": "text",
                "text": (
                    "你刚生成的方法与已经失败的方法完全相同，因此没有再次执行。"
                    "请根据复查意见提出实质不同的Pipeline；仅修改名称或说明文字不算新方法。"
                ),
            })
        human_feedback = previous_context.get("human_feedback") or {}
        if human_feedback:
            feedback_text = human_feedback.get("incremental_description")
            if feedback_text:
                content.append({"type": "text", "text": f"用户审核后的补充说明：{feedback_text}"})
            for key, label in (
                ("include_mask_path", "用户圈出的必须包含区域"),
                ("exclude_mask_path", "用户圈出的必须排除区域"),
            ):
                path = human_feedback.get(key)
                if path and Path(path).exists():
                    content.append({"type": "text", "text": f"下面是{label}。最终结果必须遵守该约束："})
                    content.append(image_content(path, label))
        for reference_mask in previous_context.get("reference_masks") or []:
            stats = reference_mask.get("stats") if isinstance(reference_mask, dict) else None
            if stats:
                content.append({
                    "type": "text",
                    "text": (
                        "实例图已提取到彩色标注模板："
                        f"{stats.get('region_count', 0)} 个区域，总面积 "
                        f"{stats.get('total_area_px', 0)} 像素，覆盖率 "
                        f"{float(stats.get('coverage', 0)) * 100:.2f}% 。"
                        "请在当前图上生成类似规模的语义标注，不要复制坐标。"
                    ),
                })
        content.append({
            "type": "text",
            "text": "已有任务上下文：" + json.dumps(previous_context, ensure_ascii=False, default=str)[:8000],
        })
        for key, label in (
            ("previous_result_image_path", "上一轮算法结果图"),
            ("feedback_image_path", "用户在画布上编辑后的反馈图"),
        ):
            context_image = previous_context.get(key)
            if context_image and Path(context_image).exists():
                content.append({"type": "text", "text": f"下面是{label}："})
                content.append(image_content(context_image, label))
        if previous_context.get("feedback_image_path"):
            content.append({
                "type": "text",
                "text": (
                    "画布反馈语义：红色标记表示误检区域，新结果应尽量删除；"
                    "绿色标记表示漏检区域，新结果应尽量补充。"
                    "优先基于previous_pipeline进行参数增量修改，并说明与上一轮的差异。"
                ),
            })
    return [{"role": "user", "content": content}]


def build_candidate_review_messages(
    target_image_path,
    description,
    candidates,
    reference_examples=None,
    acceptance_criteria=None,
):
    criteria = normalize_acceptance_criteria(
        acceptance_criteria,
        task_summary=description,
    )
    content = [{
        "type": "text",
        "text": (
            "你是工业视觉算法复查节点。请比较原图和每个候选的标注叠加图，"
            "根据用户描述和本任务的验收条件，逐条判断候选是否真的完成目标。"
            "不要输出分数，不要把Pipeline成功运行、结果非空或数量看似合理当作视觉正确。"
            "如果验收条件中的count_policy是observed_signal，observed_count只是视觉复查线索，不是硬性数量门禁；"
            "只有count_policy为exact且来源为user_explicit时，才要求组件数量严格匹配expected_count。"
            "Handbook标注图仅用于对照目标类别、边界和标注风格，不能作为当前图的像素标注。"
            "只有至少一个候选满足全部关键视觉条件时才能返回present；"
            "只要存在明显漏标、误标、边界错误或输出形式错误，就返回revise。"
            "只输出JSON，结构为："
            '{"decision":"present|revise",'
            '"selected_candidate":"候选名称或null",'
            '"reason":"简短理由",'
            '"observed_issues":["观察到的问题"],'
            '"revision_plan":["下一轮修改建议"]}'
            f"\n用户描述：{description or ''}"
            "\n本任务验收条件："
            + json.dumps(criteria, ensure_ascii=False)
        ),
    }, image_content(target_image_path, "当前待处理原图")]
    for index, example in enumerate(normalize_reference_examples(reference_examples), start=1):
        content.append({
            "type": "text",
            "text": f"甲方Handbook标注示例 {index}：{example.get('description') or '已标注参考图'}",
        })
        content.append(image_content(example["image_path"], f"Handbook标注示例 {index}"))
    for item in candidates:
        if item.get("status") not in {"completed", "selected_for_review"}:
            continue
        result_path = Path(item.get("directory", "")) / "result_annotation.png"
        content.append({
            "type": "text",
            "text": json.dumps({
                "name": item.get("name"),
                "hypothesis": item.get("hypothesis", ""),
                "facts": item.get("quality", {}),
                "measurement_summary": (item.get("measurements") or {}).get("summary", {}),
            }, ensure_ascii=False),
        })
        if result_path.exists():
            content.append(image_content(result_path, f"候选 {item.get('name')} 标注叠加图"))
    return [{"role": "user", "content": content}]


def normalize_reference_examples(raw):
    examples = []
    for item in raw or []:
        if isinstance(item, (str, Path)):
            path = Path(item)
            description = "甲方已标注的Handbook示例图"
        elif isinstance(item, dict):
            path = Path(item.get("image_path") or item.get("path") or "")
            description = str(item.get("description") or "甲方已标注的Handbook示例图")
        else:
            continue
        if not path.is_file():
            continue
        examples.append({
            "image_path": str(path),
            "description": description,
        })
    return examples[:3]


def _parse_positive_count(value):
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _extract_explicit_count(description):
    """Extract a count only from the user's explicit task wording."""
    text = str(description or "")
    match = re.search(r"(?:共|有|提取|标出|检测|识别|数量(?:为|是)?)[^0-9]{0,8}(\d+)\s*(?:个|只|枚|孔|颗粒|目标|椭圆|物体)?", text)
    if not match:
        return None
    return _parse_positive_count(match.group(1))


def _strip_observed_count_limits(pipeline, observed_count, explicit_count):
    """Do not turn a task/visual count into a hard component-selection limit."""
    count_limit = explicit_count or observed_count
    if not count_limit or not isinstance(pipeline, dict):
        return pipeline
    for step in pipeline.get("steps", []):
        if not isinstance(step, dict) or step.get("op") != "filter_components":
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        max_components = _parse_positive_count(params.get("max_components"))
        if max_components is not None and max_components <= count_limit:
            params.pop("max_components", None)
    return pipeline


def normalize_task_understanding(raw, task_description=None):
    from core.pipelines.dsl import normalize_pipeline

    value = raw if isinstance(raw, dict) else {}
    plans = value.get("candidate_plans") if isinstance(value.get("candidate_plans"), list) else []
    strategy = normalize_strategy(value.get("recommended_strategy"))
    raw_candidates = value.get("candidate_pipelines")
    candidates = []
    if isinstance(raw_candidates, list):
        for index, candidate in enumerate(raw_candidates[:3]):
            if not isinstance(candidate, dict):
                continue
            pipeline = normalize_pipeline(
                candidate.get("pipeline"),
                name=str(candidate.get("name") or f"candidate_{index + 1}"),
            )
            if pipeline.get("kind") != "builtin_pipeline":
                _require_explicit_operator_definitions(pipeline)
            candidates.append({
                "name": str(candidate.get("name") or f"candidate_{index + 1}"),
                "hypothesis": str(candidate.get("hypothesis") or ""),
                "pipeline": pipeline,
            })
    # Empty or malformed provider output is recoverable: the deterministic
    # local baselines are selected by the planner after normalization.
    rendering = value.get("rendering") if isinstance(value.get("rendering"), dict) else {}
    description_text = str(value.get("task_summary") or "").lower()
    requested_fluorescent_green = any(
        phrase in description_text
        for phrase in ("荧光绿", "fluorescent green", "neon green")
    )
    color = str(
        rendering.get("contour_color")
        or ("#39FF14" if requested_fluorescent_green else "#ff4030")
    )
    if not (len(color) == 7 and color.startswith("#")):
        color = "#ff4030"
    try:
        thickness = max(1, min(10, int(rendering.get("contour_thickness", 1))))
    except (TypeError, ValueError):
        thickness = 1
    raw_constraints = value.get("target_constraints") if isinstance(value.get("target_constraints"), dict) else {}
    explicit_count = _extract_explicit_count(task_description)
    raw_observed_count = _parse_positive_count(
        raw_constraints.get("observed_count") or raw_constraints.get("expected_count")
    )
    if explicit_count is not None:
        expected_count = explicit_count
        observed_count = raw_observed_count
    else:
        expected_count = None
        observed_count = raw_observed_count
    constraints = dict(raw_constraints)
    constraints.pop("expected_count", None)
    constraints.pop("observed_count", None)
    constraints.pop("count_source", None)
    if expected_count is not None:
        constraints.update({
            "expected_count": expected_count,
            "count_source": "user_explicit",
        })
    elif observed_count is not None:
        constraints.update({
            "observed_count": observed_count,
            "count_source": "model_observed",
        })
    for candidate in candidates:
        _strip_observed_count_limits(candidate.get("pipeline"), observed_count, expected_count)
    output_requirements = [str(item) for item in value.get("output_requirements", [])]
    if not output_requirements:
        output_requirements = ["mask", "measurements"]
    acceptance_criteria = normalize_acceptance_criteria(
        value.get("acceptance_criteria"),
        task_summary=str(value.get("task_summary") or ""),
        output_requirements=output_requirements,
    )
    if expected_count is not None:
        acceptance_criteria.update({
            "count_policy": "exact",
            "expected_count": expected_count,
        })
    elif observed_count is not None:
        acceptance_criteria.update({
            "count_policy": "observed_signal",
            "observed_count": observed_count,
        })
    requested_mode = rendering.get("annotation_mode")
    if requested_mode:
        annotation_mode = str(requested_mode).lower()
    elif "bbox" in output_requirements:
        annotation_mode = "bbox"
    elif "contours" in output_requirements:
        annotation_mode = "contour"
    else:
        annotation_mode = "mask"
    if annotation_mode not in {"contour", "mask", "bbox"}:
        annotation_mode = "mask"
    try:
        mask_alpha = max(0, min(255, int(rendering.get("mask_alpha", 72))))
    except (TypeError, ValueError):
        mask_alpha = 72
    return {
        "task_summary": str(value.get("task_summary") or ""),
        "target_defect": str(value.get("target_defect") or ""),
        "normal_context": str(value.get("normal_context") or ""),
        "ambiguities": [str(item) for item in value.get("ambiguities", [])][:10],
        "questions": [str(item) for item in value.get("questions", [])][:10],
        "output_requirements": output_requirements,
        "acceptance_criteria": acceptance_criteria,
        "candidate_plans": plans[:3],
        "candidate_pipelines": candidates,
        "target_constraints": constraints,
        "rendering": {
            "annotation_mode": annotation_mode,
            "contour_color": color,
            "contour_thickness": thickness,
            "mask_alpha": mask_alpha,
        },
        "recommended_strategy": strategy,
        "confidence": float(value.get("confidence", 0.5)),
    }


def normalize_acceptance_criteria(raw, task_summary="", output_requirements=None):
    value = raw if isinstance(raw, dict) else {}

    def _strings(items, limit=10):
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            return []
        return [str(item).strip() for item in items if str(item).strip()][:limit]

    requested_output = _strings(value.get("requested_output"))
    if not requested_output:
        requested_output = _strings(output_requirements) or ["mask"]
    visual_checks = _strings(value.get("visual_checks"))
    if not visual_checks:
        visual_checks = ["结果应覆盖用户描述的目标，并且标注位置和边界与原图一致"]
    failure_examples = _strings(value.get("failure_examples"))
    if not failure_examples:
        failure_examples = ["结果存在明显漏标、误标、边界偏离或输出形式不符"]
    return {
        "task_goal": str(value.get("task_goal") or task_summary or "完成用户描述的视觉标注任务"),
        "requested_output": requested_output,
        "visual_checks": visual_checks,
        "failure_examples": failure_examples,
    }


def normalize_candidate_review(raw, candidate_names):
    value = raw if isinstance(raw, dict) else {}
    names = {str(name) for name in candidate_names if name is not None}
    selected = value.get("selected_candidate")
    selected = str(selected) if selected is not None else None
    decision = "present" if value.get("decision") == "present" else "revise"
    reason = str(value.get("reason") or "视觉复查认为结果还需要调整。")
    if selected not in names:
        selected = None
        if decision == "present":
            decision = "revise"
            reason = "视觉复查没有指出可直接展示的有效结果。"

    def _strings(items):
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            return []
        return [str(item).strip() for item in items if str(item).strip()][:10]

    return {
        "decision": decision,
        "selected_candidate": selected,
        "reason": reason,
        "observed_issues": _strings(value.get("observed_issues")),
        "revision_plan": _strings(value.get("revision_plan")),
    }


def _require_explicit_operator_definitions(pipeline):
    from core.operators import build_default_registry

    generated_names = {
        str(item.get("name"))
        for item in pipeline.get("generated_operators", [])
        if isinstance(item, dict) and item.get("name")
    }
    allowed_names = set(build_default_registry(pipeline.get("generated_operators", [])).names())
    undeclared = sorted({
        str(step.get("op"))
        for step in pipeline.get("steps", [])
        if isinstance(step, dict)
        and step.get("op") not in generated_names
        and step.get("op") not in allowed_names
    })
    if undeclared:
        raise ValueError(
            "candidate pipeline uses unknown operators: "
            + ", ".join(undeclared)
        )


def extract_json_object(text):
    content = (text or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content
        content = content.rsplit("```", 1)[0].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Vision provider did not return a JSON object")
    return json.loads(content[start : end + 1])


def image_content(path, label):
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
            "detail": "high",
        },
    }
