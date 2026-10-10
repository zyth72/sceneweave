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

## 第二轮：event intensity sweep + role sensitivity

`scripts/probe_system1_sensitivity.py` 是独立于第一轮的敏感度探针，仍只调用 System1、只输出三路倾向分布
（`continue/replan/rethink` 之和为 1）与 `note`；argmax 仅用于离线展示/评分，不抽样、不制定正式路由策略。
它复用第一轮的严格解析、无工具 OpenCode 调用与 usage 补取，不复制其整份实现。

三组控制状态（共 12 个，`scripts/probe_system1_sensitivity_cases.json`）：

- `intensity`（6 级）：同一意向/保存计划/当前状态，只变收件人的已感知移动范围（从原位置移动约 0.5 米仍在伸手
  可及范围 → 约 2 米 → 房间另一侧 → 离开房间但去向已知 → 离开建筑但去向已知 → 去向未知）。**不硬规定单调**，
  只观察趋势与符号变化。最低级明确写“移动 0.5 米、仍在伸手可及范围”，不是未移动。
- `influence`（3 版）：同一共同任务前提下的三种强度/层次事件——巨响但已确认无关且安全（期望 continue）、
  原路径堵死但有已知替代路线且目标保留（期望 replan）、原交付目标被明确取消（期望 rethink）。检查有无
  “越强越 rethink”的错误泛化。
- `unconfirmed`（3 态）：抓取结果暂未确认且确认反馈待来（`pending`：允许 continue/replan，不得误当失败/完成或
  强制 rethink，也不得据此提前提交依赖 a2）；抓取已实际 `completed` 且缓存交付可提交（此时才可提交依赖它的
  a2，对照 continue）；抓取已 `failed` 且知道另有可行做法（对照 replan）。三组中的未来 a2 都是内部保存、未提交
  的后续动作（saved），不是 registered；已结束的 a1 各自保留 unconfirmed/completed/failed 状态。

真实角色来源（用户已明确授权，只读）：`/home/zyth/lin/worldbook/persona/{贝尔法斯特,企业,恶毒}.json`。
JSON 会整体读入以校验结构，但只把固定白名单字段（`personality` 的核心特征/优点/缺点（性格缺陷）/习惯或怪癖、
`worldview` 的道德准则、`innerDrive` 的核心动机/短期目标/恐惧与禁忌）抽取并注入 prompt；**不抽取或注入**任何
非白名单字段（外貌/服装/三围/爱情/亲密台词/关系底账等），不写 `~/lin`。抽取文本与来源 sha256 写入输出目录
`manifest.json`；模型看到的是中性编号（P1/P2/P3）而非真实姓名，报告再对应真实姓名。默认再加一份无人格基线
（其余上下文相同），避免只凭知名角色名套模板。

```bash
# 离线：只生成 prompt 与任务数
uv run --locked python scripts/probe_system1_sensitivity.py --dry-run

# 小烟测：每 suite 首个状态 × baseline+P1 × 1 次
uv run --locked python scripts/probe_system1_sensitivity.py --smoke

# 完整：12 状态 × 4 profile × 2 次 = 96 次（分布模式）
uv run --locked python scripts/probe_system1_sensitivity.py
```

可选参数：`--suite intensity|influence|unconfirmed|all`、`--profile`（可重复，支持编号或真实角色名）、
`--repeats`、`--limit`、`--model`、`--agent`、`--opencode`、`--workdir`、`--persona-dir`、`--output-dir`、
`--usage-db`、`--no-db-usage`、`--timeout`、`--seed`、`--no-thinking`、`--dry-run`、`--smoke`。

已有记录可离线重导出 Markdown，无需重新调用模型，也不读取当前人格仓库或用例文件：

```bash
uv run --locked python scripts/probe_system1_sensitivity.py \
  --report-from sceneweave-runs/system1-sensitivity/20261010-corrected
```

此命令只更新 `report.md` 和派生统计 `summary.json`，保留原始 `manifest.json`、`samples.jsonl`。
历史场景来自保存的 manifest，提示词和输出来自逐次记录，不按当前模板重建。未保存的 OpenCode／供应商
附加系统提示不在记录范围内，不能把 CLI 输入文本等同于完整供应商请求。

输出默认写入忽略目录 `sceneweave-runs/system1-sensitivity/<timestamp>/`：`manifest.json`（完整的受控用例：
事件/意向/计划/已知行动状态/认知/离线期望与理由，以及模型/suite/repeats/seed/timeout 等参数和用例文件、
脚本与人格来源的 sha256）、`samples.jsonl`（完整 prompt/原始 response/events/model/sessionID/usage/耗时/error）、
`summary.json`、`report.md`。Markdown 首先提供场景目录、人格摘录与来源、每个场景的完整事件／意向／保存计划／
已知行动状态／记忆，以及各人格实际完整提示词、全部重复的原始响应、note、保存的 reasoning、task/session ID、
实际调用顺序及失败信息。提示词仅在同场景同人格的重复间逐字一致时合并展示；不同提示词分别列出。
合成场景与真实人格来源分开说明，不补写未提供的剧情时刻或空间事实，离线参考标签明确标为未发给模型。
报告按 profile 与 suite 都列出耗时 median/p95、tokens（含 cache）与 cost、
usage 覆盖；给出**逐 profile 的 6 级趋势**（不只看混合全部 profile 的池化），并同时给出实际/有效/预期样本覆盖
与分组缺失记录；同状态重复波动用每路径 **max-min（全重复范围）**，基线差与两侧范围同屏比较；含邻级与同强度
不同层比较、逐条错误。输出目录非空时拒绝覆写混入旧样本；任一调用/schema/工具失败时脚本以非零状态退出。

## 第二轮限制

- 三路数值是模型**自报倾向**，未经校准，不是世界事件概率或真实决策概率。
- 真实姓名替换为中性编号，但角色设定内容可能仍被模型凭已有知识识别，属**非完全盲测**。
- 各 profile 抽取文本长度不同，模型对不同角色的既有知识也不同，二者与人格设定混杂，不能单独归因。
- 小样本不宣称统计显著；profile−baseline 差值须与两侧全重复 max-min 范围同屏比较，只描述现象。
- cost 是 OpenCode usage 返回的元数据，不是账单。

## 第二轮实测（2026-10-10）

修正批位于本机忽略目录 `sceneweave-runs/system1-sensitivity/20261010-corrected/`；`report.md` 保留完整统计，
`review.md` 是人工复核。共 96 次独立调用，96/96 有效、48/48 组完整，无调用／schema／工具错误或并列最大。
已离线核对任务覆盖、严格解析、来源哈希及各人格条件使用相同任务上下文。原始人格摘录和模型输入不进 Git。

- 三种影响层级共 24/24 符合参考：无关安全巨响 continue、原路堵死但有替代路线 replan、交付目标取消 rethink。
- 未确认等待、确认完成都偏 continue，失败且有其他做法偏 replan（另 24/24）。note 明确区分等待确认；
  这只验证 System1 分流，依赖是否允许提交仍由运行时处理。
- 0.5 米到 2 米时，各条件 continue 均值下降、replan 上升；其后已知去向的各级主要偏 replan，但并不单调。
  因此支持粗粒度边界，尚不能证明连续强度编码或概率校准。

去向未知时的两次均值（顺序 continue / replan / rethink）：

| 条件 | continue | replan | rethink | 两次最大倾向 |
| --- | ---: | ---: | ---: | --- |
| 无人格基线 | 0.100 | 0.800 | 0.100 | replan / replan |
| 贝尔法斯特 | 0.050 | 0.325 | 0.625 | rethink / rethink |
| 企业 | 0.100 | 0.700 | 0.200 | replan / replan |
| 恶毒 | 0.035 | 0.290 | 0.675 | rethink / rethink |

贝尔法斯特、恶毒的 rethink 相对基线均值差为 +0.525、+0.575，双方在本批的重复范围分别为 0.150、0.050
（基线为 0），值得复测，但两次重复不能证明统计显著或单独归因于人格。部分解释把位置未知说成对象消失或目标
失效；这不是已成立的事实，也提醒我们继续区分目标与寻找／等待／上报等做法，不能把角色差异直接等同正确性。

重复 argmax 一致 47/48 组；唯一不一致是贝尔法斯特的 0.5 米案例，0.35/0.55/0.10 与 0.82/0.16/0.02，
两次分别偏 replan、continue。continue 的人格均值差 +0.310 小于自身重复范围 0.470，不能当稳定人格差异；
其解释在固定坐标与收件人实体两种绑定间变化，测试计划仍有这处歧义。

CLI 耗时中位数 4.960 秒、p95 7.352 秒，usage 覆盖 96/96。OpenCode cost 元数据合计 $0.022825632（非账单）；
input 55488、output 9791、reasoning 14221、cache_read 31744、cache_write 0，不把缓存忽略后当总 token。
正式路由仍不改变。先前的 `20261010-124237` 与旧编码会话重复启动的 `20261010-124817` 均因测试状态矛盾中止；
保留原始记录，在各目录 `EXCLUDED.md` 说明，不与修正批混合，也不作最终结论。

## 第三轮：相同输入的 route 字段与三路倾向分布配对

`scripts/probe_system1_comparison.py` 是独立的第三轮探针：在**同一敏感度输入**下同时问两种输出任务，
比较 `field`（严格枚举 route 字段）与 `distribution`（三路倾向分布）是否给出相同的路径倾向。
它仍只调用 System1，不启动 System2/System3/翻译器/World/engine，不生成动作，不影响正式路由。

来源是第二轮的只读记录（默认 `sceneweave-runs/system1-sensitivity/20261010-corrected`）：
从保存的 `manifest.json` 读取真实人格摘录与 12 个合成状态，从 `samples.jsonl` 读取原始完整 prompt。
**不读取当前 `~/lin`，也不按当前模板重建** prompt；原始记录只读。按原 prompt 的最后一个独立
`【输出】` 段切成共同前缀与输出指令：`distribution` 保留旧完整提示词原文，`field` 在同一前缀后
换成固定的 `{"route": "continue|replan|rethink", "note": "简短说明"}`。同一 (case, profile, repeat)
的两条调用因此共享逐字一致的事件/人格/意向/计划/状态/记忆/三路定义，只【输出】段不同，前缀 sha256
写入 manifest 与报告可逐条核对。来源缺少标记、覆盖不全、身份重复、同 case/profile 重复 prompt
不一致时明确报错，不暗中取第一条。

每对 (case, profile, repeat) 是一种 pair：12 状态 × 4 profile × 2 次重复 × 2 模式 = 192 次全新独立调用。
固定 seed 打乱 pair 顺序，并为每 pair 随机决定 field/distribution 先后，两条连续执行；seed 只控制
实验顺序与两模式先后，**不声称设置了模型真实温度或 seed**。旧 96 次只作为输入来源，绝不混入本轮样本，
也不假装是新鲜配对。

```bash
# 离线：只生成 prompt 与任务数
uv run --locked python scripts/probe_system1_comparison.py --dry-run

# 小烟测：每 suite 首例 × baseline/P1 × 每模式 × 1 次 = 12 次
uv run --locked python scripts/probe_system1_comparison.py --smoke

# 完整：12 状态 × 4 profile × 2 次 × 2 模式 = 192 次
uv run --locked python scripts/probe_system1_comparison.py
```

可选参数：`--source-dir`、`--repeats`、`--model`、`--agent`、`--opencode`、`--workdir`（默认
`/tmp/opencode/system1-comparison`）、`--output-dir`、`--usage-db`、`--no-db-usage`、`--timeout`、
`--seed`、`--no-thinking`、`--dry-run`、`--smoke`、`--report-from`。沿用无工具 agent、严格解析、
真实 elapsed 与缺失 `step_finish` 时按 sessionID 从会话库补取 usage；每次新会话 standalone。

输出默认写入忽略目录 `sceneweave-runs/system1-comparison/<timestamp>/`：

- `manifest.json`：完整 case/persona 摘录、来源路径与文件 sha256、模型/repeats/参数、明确的
  `task_order`（pair_id 与 mode 顺序）、共同前缀 sha256、两种完整 prompt。
- `samples.jsonl`：逐次立即 append——pair_id、case/profile/repeat/mode、完整 prompt、原始
  text/note/reasoning/events、model/sessionID、调用/schema/tool/exit/timeout 结果、usage 来源/tokens/cost。
- `summary.json` / `report.md`：见下。

报告至少含：每模式预期/实际/有效覆盖、完整可比较 pairs、配对一致/不一致/缺失/失败（分母明确）；
分别给出 field schema 合法率、distribution 合法率及无并列可评分率；每模式同 case/profile 重复的
路由稳定性（至少两次且全重复有效才比较，失败不计为稳定）；每模式耗时 median/p95、tokens 各原始字段
（含 cache）、usage 覆盖/来源、cost 元数据，以及配对耗时差 field−distribution 的 median；
参考标签符合率（失败计入预期分母，有效分母只含可评分样本；intensity 无固定答案不虚报正确率）；按 case/profile
逐 pair 并列表（field route/note、distribution 三值/argmax/note 含所有不一致）；逐场景完整状态、
4 人格来源与实际摘录、两种完整实际 CLI 提示词、每 pair 两种原始响应与 note、可折叠 reasoning、
task/session/调用序号/error。提示词只在逐字相等时合并，引用 `sens._fenced` 避免原文围栏破坏 Markdown；
旧批次明确标为只读输入来源，不与本轮样本混合。输出目录非空时拒绝覆写；任一调用/schema/tool/覆盖/pair
丢失或失败时脚本以非零状态退出。

已有记录可离线重导出 Markdown，不读取来源、不初始化 agent、不调用模型，且不动原始文件：

```bash
uv run --locked python scripts/probe_system1_comparison.py \
  --report-from sceneweave-runs/system1-comparison/<timestamp>
```

`tests/test_system1_comparison.py` 覆盖前缀分割与唯一输出段差异、来源身份覆盖与重复 prompt 歧义拒绝、
seed 打乱与 pair 顺序、两种严格 schema 与并列、失败/缺失分母与稳定性、原始输入输出完整进入 Markdown、
离线重导出不改原始文件且不访问来源/API、非空输出拒绝与 timeout 范围。默认测试用合成来源与假模型，
**绝不**进行真实模型调用，也不读取真实人格仓库。

### 第三轮限制

- 两种输出任务格式不同（枚举字段 vs 三路分布），差异可能来自任务格式本身，不能单独归因于模型能力。
- 三路数值是模型**自报倾向**，未经校准，不是世界事件概率或真实决策概率；argmax 不是正式路由策略。
- 小样本不宣称统计显著；真实姓名虽替换为中性编号，角色设定内容仍可能被模型凭已有知识识别，属非完全盲测。
- seed 只控制实验顺序与两模式先后；无法设置模型真实温度/seed。cost 是 OpenCode usage 元数据，不是账单。
