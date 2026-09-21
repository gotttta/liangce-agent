# 系统记忆

## 已实现的边界

短期上下文由 `core/memory/context.py` 组装。长期记录存在
`<TaskStore.root.parent>/memory.sqlite3`，图像、Mask、Pipeline 和 Skill 继续使用原有文件。

- 语义：当前目标、有效约束、模型观察。记录来源、状态及版本。
- 程序：验收算法的索引、适用特征及原算法路径。检索继续由 AlgorithmRegistry 执行；SkillRegistry 保持原有加载方式。
- 情景：消息、任务事件、每轮结果、实验引用和上下文装配清单。

TaskStore 是统一业务入口。Web 在视觉理解前准备记忆；CLI/直接图调用也会创建或继续任务。
`run_agent_graph(..., task_store=store, task_id=...)` 可继续同一任务。
CLI 支持 `--task-id`，输出包含 task_id 和 graph_thread_id。

## 写入与可信程度

模型通过 `memory_updates` 返回 set/revoke 操作；只允许修改当前任务的 current_goal 或
constraint:*，必须携带本轮用户消息中的原文片段。目标不能被 revoke。程序检查字段、范围、来源片段，
但这不是对自然语言含义的确定性证明；仍需评测模型是否正确解读否定和修改意图。
`target_constraints` 中的模型观察只进入 hypothesis，不自动成为用户事实。
自动修订轮次不能再次修改用户语义事实。

遗漏旧约束不表示撤销。修改会使旧版本成为 superseded，撤销保留 revoked 版本。
当前范围默认是任务；产品/项目层的实体身份、跨范围继承和语义向量检索尚未开放。

已有 memory.json 在首次读取时迁移为带 legacy 来源的假设，不把旧推断标为用户确认。
memory.json 继续写兼容快照。原 conversation.jsonl、事件和产物不搬迁。
SQLite 与兼容文件不是同一事务；SQLite 是新增语义记录的事实来源，文件仍是原始消息/产物来源。

## 上下文

短期记忆采用三部分：**最近 4 轮原文 + 更早对话滚动摘要 + 独立有效约束**。

- 一轮从一条用户文本消息开始，包含其后的助手回复，直到下一条用户文本消息。
  先过滤纯 UI 进度卡片和非文本附件，再划分轮次；近期业务文本保留原文，不逐条截断。
  图片由视觉请求独立加载，不把图片 base64 写进对话摘要。
- 更早历史由当前 Aliyun provider 进行独立的纯文本摘要调用。输入是旧摘要和刚移出近期窗口的消息，
  不反复发送全部旧历史。默认摘要不超过 4,000 字符，每批新增历史约 16,000 字符；单条超长消息完整处理。
  摘要要求保留目标变化、用户修正/撤销、关键决策、实验结论和未解决问题，并区分用户要求、模型推测与事实。
- 摘要保存在 SQLite 的任务级情景记录 `conversation_summary`，记录已覆盖消息数量、历史前缀哈希和版本。
  同一历史重复调用不会再次摘要；服务重启可以续用；已覆盖历史被修改或缩短时重新构建。
- 读取完整持久化对话，只有不存在持久化消息时才使用 UI 历史；UI 不再提前截断为 12 条。
- 目标、有效约束及来源、人工反馈独立加载。摘要是历史背景，不是有效约束的事实来源，
  不能恢复已撤销的要求，也不能直接写入语义事实。

以上三部分优先保留，附加实验等信息按预算整段保留或省略。清单记录 included/omitted、memory_ids、
summary_covered_messages 和 recent_turns。默认仍是 16,000 字符预算，不是精确 tokenizer 预算；
必带内容超额时保留并标记 mandatory_over_budget，不静默截掉近期原文或有效约束。
视觉 provider 继续使用已有实验修订摘要逻辑，因此该预算不是完整模型请求的 token/图像上限。

摘要请求失败、返回空文本或超限时，保留上一次有效摘要及尚未压缩的原文，本轮仍可继续；
不推进已覆盖位置，下轮重试。失败路径可能超出字符预算，日志会记录摘要失败。
没有摘要能力的 provider/直接 prompt 构建调用使用相同轮次划分，并保留未压缩历史作为降级输入。
摘要本身仍可能遗漏细节，完整原文保留在 conversation.jsonl 中供追溯。

换图按内容哈希清理上一轮像素反馈、结果图和基线。任务范围的文字约束仍保留；
图片范围的业务规则应显式更新。相似历史算法仍需要在当前图片重新验证。

## 执行恢复

使用 langgraph-checkpoint-sqlite 3.0.3。默认位置 `workspace/checkpoints.sqlite3`，
可用 LIANGCE_CHECKPOINT_PATH 指定。通过同一 graph_thread_id 恢复人工暂停；
run_agent_graph 使用已存在的 thread_id 时返回待验收/已完成状态，或继续未完成节点。
输入内容或请求变更必须创建新运行。同一运行的图状态和修订预算随 checkpoint 保存。
CLI `--thread-id` 可用于按原输入、原请求恢复运行。

检查点保证已完成节点不会因正常恢复重跑，但不提供文件/沙箱副作用的 exactly-once 保证：
若进程在节点内部产生结果后、提交 checkpoint 前崩溃，该节点仍可能重跑。
实验级 call_id 事务与跨运行预算管理是后续独立工作。

## 验证

`tests/test_conversation_memory.py` 覆盖近期原文、增量摘要、持久化隔离、失败降级、历史变更失效和真实 prompt 注入。

`tests/test_memory.py` 覆盖：来源校验、目标修改、约束保留/撤销/隔离、旧数据迁移、
换图清理、上下文必带要求、真实 provider prompt、跨进程 checkpoint 恢复，以及图入口重复恢复。

暂未接入 Mem0、embedding 或自动生成 Skill。后续可在 MemoryService 后增加召回适配器，
不改变事实来源、验收边界和现有产物位置。
