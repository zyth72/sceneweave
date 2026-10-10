# System1 路由探针

独立于真实模拟的实验脚本，只调用 System1，比较两种输出表达哪一种更好用：

- `field`：直接输出 `{"route": "continue|replan|rethink", "note": "..."}`。
- `distribution`：输出三路**倾向分布** `{"probabilities": {"continue": .., "replan": .., "rethink": ..}, "note": "..."}`，由程序离线取唯一最大值。

## 语义边界

- 分布模式表达的是「这次事件对角色当前状态的影响所驱动的路径倾向」，**不是**世界事件发生概率，也不是随机抽签，更不是已校准的置信。
- 探针只做 System1 单点判断，**绝不**启动 System2/System3/翻译器/World/engine 完整循环，不生成意向文本或动作，不影响正式模拟路由。
- 输入只包含该次事件与本角色私有状态（当前意向、已保存计划/行动、近期认知），不包含其他角色私有信息，也不包含参考答案。
- 程序接收分布后怎么路由，留待用户后续决定；离线 argmax 评分只是对人工标签的比对，**不是**正式程序路由策略。

## 调用方式

调用通过已配置的 OpenCode CLI 完成（默认 `/usr/sbin/opencode`、模型 `opencode-go/deepseek-v4.1-flash`），在临时独立目录、无工具 primary agent、每次独立新会话中进行；不读取 `.env`/密钥，不连接常驻服务（`--standalone` 私有服务器）。

```bash
# 只生成待发 prompt 与任务数，不做外部调用
uv run --locked python scripts/probe_system1_routing.py --dry-run

# 先跑 2 案例 × 2 模式 × 1 次，检查调用链
uv run --locked python scripts/probe_system1_routing.py --smoke

# 完整：12 案例 × 2 模式 × 2 次（默认）
uv run --locked python scripts/probe_system1_routing.py
```

可选参数：`--repeats`、`--case`、`--limit`、`--mode field|distribution|both`、`--timeout`、`--model`、`--agent`、`--opencode`、`--workdir`、`--output-dir`、`--seed`、`--no-thinking`、`--dry-run`、`--smoke`。

用例在 `scripts/probe_system1_cases.json`：9 个明确案例（三路平衡）+ 3 个边界案例（允许多个合法路由），只含私有上下文与该次事件；`expected_routes` 仅供离线评分，不进入模型提示。

## 输出

默认写入忽略目录 `sceneweave-runs/system1-probe/<timestamp>/`：

- `samples.jsonl`：逐次调用记录——原始事件（去重后）、输入 prompt、文本/推理输出、sessionID/model/退出码、是否超时、是否工具污染、真实 elapsed 秒、usage 原始字段（`input`/`output`/`reasoning`/`cache.read`/`cache.write`、`cost`、`finish`）与来源（`step_finish` 或极短响应时按 sessionID 从会话库精确补取的 `session-db`）。
- `summary.json`：合法 schema 率、准确率（失败计入分母）与有效样本准确率、分模式/分类型混淆、重复一致性、概率 margin/entropy、耗时中位数/p95、tokens/cost 原始汇总与 usage 来源。
- `report.md`：以上内容的可读报告与逐条错误。

`seed` 只控制实验顺序与两模式交错；**无法设置模型真实温度/seed**，模型侧仍有随机性。极短响应可能不产生 `step_finish`，此时脚本按本次生成的 sessionID 从 OpenCode 会话库只读补取 usage（只取 tokens/cost/finish/model，不读取或转述对话内容）；仍取不到则记为不可用，不做估算。小样本不能据此宣称校准或统计显著。

## 离线测试

`tests/test_system1_probe.py` 覆盖严格解析、并列最大记 tie、去重、工具污染标记、任务编排与 dry-run；默认测试**绝不**进行真实模型调用。重复一致性只统计至少两次且全部有效、可唯一评分的组，失败不计为稳定；缺失 token 字段标为不可用。自定义工作目录已有不同配置时拒绝覆盖。

## 首次实测（2026-10-10）

通过 OpenCode 的 `opencode-go/deepseek-v4.1-flash` 完成 12 例 × 2 模式 × 2 次，共 48 次独立 System1 调用，没有后续模块或工具调用。完整报告位于本机 `sceneweave-runs/system1-probe/20261010-120131/report.md`，逐次原始输入、输出与 usage 在同目录 `samples.jsonl`。另有先期烟测记录，不合并进这48次。

| 指标 | route字段 | 三路倾向分布 |
| --- | --- | --- |
| 输出结构合法 | 24/24 | 24/24 |
| 明确案例符合参考标签 | 18/18 | 18/18 |
| 边界案例落入允许路由集合 | 6/6 | 6/6 |
| 重复选择一致 | 11/12组 | 11/12组 |
| CLI墙钟耗时中位数 | 4.52秒 | 4.73秒 |
| 输出token合计（OpenCode usage） | 1069 | 2113 |
| reasoning token合计（OpenCode usage） | 6919 | 3020 |

两种表达在本批案例的路由评分没有差别，都是含糊响动案例出现一次continue、一次replan。分布多提供了路径权重：例如一次响动输出0.55/0.35/0.10，而无关脚步声输出0.95/0.04/0.01（顺序continue/replan/rethink）。它便于程序后续制定策略，但本探针没有验证任何正式路由策略，也没有证明这些数字已校准。

24次每模式的小样本中，分布模式输出更长，reasoning字段更少；完整usage、cache和耗时分模式保留在报告中。这不足以推断长期更省token或更准确。边界案例允许多个路由，因此“符合标签”也不等于有唯一正确答案。
