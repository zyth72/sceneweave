# sceneweave

使用 uv 管理、基于 Google DeepMind Concordia 的角色模拟项目。

- [核心循环与角色职责说明](docs/核心循环与角色职责说明.md)
- [模块交互契约](docs/模块交互契约.md)
- [开发规范](docs/开发规范.md)
- [项目协作要求](AGENTS.md)

## 运行环境

Python 3.14，uv，Concordia 2.4.0（发行包 `gdm-concordia`）。依赖版本由 `uv.lock` 锁定。

```bash
uv sync --locked
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy
uv run --locked pytest
```

## 代码放在哪里

```text
src/sceneweave/
├── domain/
│   ├── identity.py               # 跨模块身份类型与非空校验
│   ├── time.py                   # 非负整数毫秒与秒配置入口转换
│   ├── messages.py               # 周期输入、响应、打回与检查结果消息
│   ├── translation.py            # 行动翻译输入输出消息（新行动/修改安排）
│   ├── adjudication.py           # 裁决候选、行动节点、结算结果与进展更新
│   ├── scheduling.py             # 引擎事件类型与按最近时刻跳转的事件队列
│   └── tracing.py                # 领域追踪接口（不依赖 Concordia）
├── world/base.py                 # World 接口：进展更新、检查、裁决、提交与发布
├── world/scene.py                # 不可变场景事实模型、场景卡加载与结构化补丁
├── world/model_world.py          # 模型驱动 World（作用域采样、原子提交、可见性发布）
├── engine/
│   ├── config.py                 # 运行配置：内思考间隔与最长时长（秒入口转毫秒）
│   ├── scheduler.py              # 剧情时钟、事件调度、翻译登记与运行编排
│   ├── actions.py                # 行动登记记录与统一登记计划（先校验后提交）
│   ├── registration.py           # 登记前时间安排纯校验（新行动与修改）
│   ├── responses.py              # 一次处理周期的全角色响应与检查/打回屏障
│   ├── report.py                 # 单周期报告：角色可见视图与客观/诊断内容
│   └── control.py                # 运行门协议与运行状态（供控制适配注入）
├── roles/base.py                 # 角色 perceive/rejected 交互契约
├── roles/model_role.py           # 模型驱动角色：内部串行编排 System1/2/3
├── roles/model_output.py         # 模型 JSON 输出的严格解析（错误不转空）
├── roles/components/             # System1/2/3 组件与采样器协议、阶段留痕
├── translator/base.py            # 行动翻译器可注入契约
├── translator/model_translator.py # 模型驱动翻译器（作用域采样装配、严格结构校验）
├── config/loader.py              # 严格文件配置加载（场景/角色/事件/运行参数）
├── app/                          # 统一装配、脚本假模型、运行编排与 UI 页面
│   ├── assembly.py               # 从配置构建 Scheduler 装配（live/scripted）
│   ├── scripted.py               # 脚本假模型（从示例提取）
│   ├── runner.py                 # headless 运行、输出管理、录制/重放
│   ├── recording.py              # 录放清单、输入绑定与全部作用域消费校验
│   ├── ui.py                     # 真实 UI 生命周期、周期广播与录放
│   ├── ui_snapshot.py            # UI 只读历史/模型调用快照与固定端点
│   └── page.py                   # UI 页面资源（textContent 渲染、历史视图）
├── __main__.py                   # CLI 统一入口
└── integrations/concordia/
    ├── entity.py                 # 框架实体接入
    ├── tracing.py                # 领域追踪接入 Concordia 结构化/组件日志
    ├── sampling.py               # 语言模型采样桥（真实组件日志）与录制/重放
    ├── chat_completions.py       # OpenAI 兼容 Chat Completions 真实工厂（--live 内置）
    └── control.py                # StepController 运行门与真实 StepData 转换
tests/                          # 不依赖真实模型的测试
examples/                       # 场景组装与演示
```

**World** 保存客观事实，并提供对完整主动行为声明的无副作用检查入口 `check_response`（携带本角色由引擎提供的只读既有安排快照；只产生记录共同状态版本的 `ActionCheckResult`，不调用角色、不提交事实、不取消行动、不把登记当作实际开始）。事件与行动的落地拆成明确边界：`adjudicate` 产生联合裁决候选，`validate_adjudication` 做运行时校验，`commit_adjudication` 原子提交实际状态变化，`publish` 才生成按角色范围过滤的 `Delivery`。具体事实模型与语义判断由实现注入，角色不能拿到 World 的可变状态引用。

**引擎** 保存剧情时钟、事件队列及已登记的行动。`Scheduler`（`engine/scheduler.py`）实现完整闭环：按最近待处理事件跳转剧情时刻（整数毫秒），同刻事件整批交给 World 联合裁决；时钟进入事件时刻后先调用 World 的进展更新（按流逝的剧情时间推进实际行动与被动过程，实际变化原子应用并递增共同状态版本），再走「候选 → 运行时校验 → 原子提交 → 发布感知」（候选基于预裁决版本：裁决与校验都不得改变状态，提交版本必须严格递增且等于实际版本，后续事件身份与结算结果在事实提交前预检；publish 与诊断快照是只读边界，调用前后版本必须一致，进展更新版本不得回退），随后复用 `ResponseRunner` 采集全部角色的完整声明（只做输入校验、并行/串行与身份唯一，不做检查），屏障之后对每个角色先翻译完整链、再由 World 对整链与完整排程做无副作用检查、检查通过后角色保存完整链与排程并提交最近步骤，最后统一登记；翻译结束后复核共同状态版本未被改动。整轮完成前剧情时间冻结：模型调用与现实等待不推进时间，引擎也不凭意向更新客观状态。行动登记由 `engine/actions.py` 的登记记录与统一登记计划承担——`prepare_registration` 是可直接调用的纯校验入口（翻译关联与类型、新行动及提交组归属、跨角色行动身份唯一、终态不可复用，并复用 `registration_issues` 对每份最终声明完整重跑结构/引用/提交组/时间/历史校验，不复制业务规则），登记计划与队列候选变更（内部事件身份避让外部事件身份）完整准备并校验后，先让修改与撤回的客观效果经 World 结构化裁决提交（修改输入携带归属、原描述、明确变化与新旧未来安排），再统一应用，失败不留下半提交状态。维持保留原安排；遗漏撤回主动维持（未开始撤销未来执行事件，执行中交 World 裁决客观效果）；修改只调整未来节点并在 World 实际生效；未确认（UNCONFIRMED）表示结果未知而非已知终止，后续节点与主动维持不凭未知删除；终态身份不可复用。周期性内思考从起始时刻按 `RunConfig` 间隔触发并与同刻外部事件合并；时限包含边界，跳到无事件的时限边界时按最后流逝剧情时间更新实际进展后停止（不结算未来、不调用角色），越界事件保留不结算。自然结束、时限、用户中止与运行错误分别明确；运行错误或周期中途取消（事件批已取出未完成）的实例禁止再次推进，门控等待处的取消未取批、可再次运行；错误现场可导出 JSON 快照（周期、已取出批、队列、行动记录、阶段、世界版本、候选与未生效计划）。`engine/registration.py` 提供登记前纯校验并把问题分为结构不一致（运行错误）与时间约束（打回）。固定的返回顺序只用于稳定呈现，不代表同刻裁决规则。

**行动翻译器** 通过 `translator/base.py` 的可注入契约接入：输入携带角色的完整候选链（尚未通过检查）、当前时刻、共同状态版本、该角色既有行动安排，以及 World 提供的只读客观上下文（`ObjectiveContext`，含共同版本与位置/姿态/障碍/实际进展等条件，不进入角色感知）；输出完整链每个动作的平坦时间安排。`ModelTranslator`（`translator/model_translator.py`）是生产可调用的模型驱动实现：经注入采样器调用模型（`shared` 单采样器 / `scoped` 按作用域装配——每角色只构造一次采样器并缓存，配合 `RecordingDirectory`/`ReplayDirectory` 可重放并行运行；同角色并发请求（含跳过路径）是明确错误且占用经 try/finally 可靠释放、不同角色并行），提示明确区分角色主观意向与 World 客观上下文，缺客观上下文是明确错误（不猜默认延迟）；输出按严格 JSON schema 解析（重复键/未知字段/bool 或浮点毫秒/重复安排身份/多余安排都是运行错误且逐类留下错误事件，不伪空、不拆组、不自行重写）；结构化链每个动作只约束自己的 start/expected_end/checkpoints（可交叠或链式，无组内共同交集要求），已有现行安排的动作不重译；时间业务约束仍由引擎校验并经打回重试，本入口不把业务时间错误转成运行错误。仅维持或空声明的轮次在翻译器入口也直接合成空安排并显式跳过留痕，不调用模型；缺客观上下文在模型调用前明确失败。阶段留痕（提示、成功链完整原始输出与逐项安排（身份/起止/节点）、跳过与各类错误（预检/工厂/采样/解析/构造/取消），携带周期/角色/响应/行动/世界版本/剧情时间）写入同一 SimulationLog；成功链追踪失败传播、错误链追踪失败不掩盖原始异常。生产模型后端参数与提示工程未在真实 API 上验证（测试与演示使用脚本假模型）。

**角色** 保存自身认知、意向和内部状态。`Role` 抽象类实现 `perceive(TurnInput)` 与 `rejected(Rejection)` 交互契约：前者接收本轮输入并返回完整主动行为声明，后者接收 World 的整链检查打回反馈并重新落实。`ModelRole`（`roles/model_role.py`）以注入的采样器驱动内部串行的 System1/2/3（同角色由角色锁串行、跨角色并行）：System1 只看本角色私有状态（当前意向与已保存计划摘要），输出三路分流 `continue`（沿用已保存完整链与排程，不调用 System2/3/翻译器）/`replan`（保留意向只重调 System3）/`rethink`（重选意向后再落实）；System2 只产当前意向（不再有 abandon 字段）；System3 输出一条**一维全量结构化原子动作链**（固定 schema 的原子动作，可选 `depends_on`；无 maintain/modify/new_actions 分类，无 feasible/retry/reason 包装，无自由文本动作）。System3 只有明确判断当前意向在当前已知条件下不可落实时，才输出独立的 `{"error":"intention_unfulfillable","reason":...}` 控制对象（不是动作、不是 World 事实）；World 整链检查拒绝先反馈 System3（保留意向重落实），只有该独立信号才升级 System2。模型不得自造真实身份：新动作无 id 时由角色一次性分配，已有动作必须携带其确认的真实 id 与完整参数；同计划内相互依赖用局部 `key` 引用后解析为真实身份；未知/终态/跨角色 id、ID-only 对象、未知字段、重复 JSON 键、空/混合类型、自依赖/重复依赖/循环依赖都明确报错。检查打回只重调 System3（保留已选意向）。计划候选只在当前步骤通过整链检查并完成统一登记后确认，替换已接受计划；打回或解析错误不提前撤销已接受计划，已完成历史不改写。角色内部保存完整链与翻译排程，对外只提交最近步骤的原子动作；依赖未满足的后续动作仍进入整链翻译与检查以获得候选排程，但不进入执行提交与登记。模型 JSON 输出经 `roles/model_output.py` 与 `domain/action_plan.py` 严格校验。每份响应的身份在阶段开始前分配并贯穿各阶段留痕；打回反馈只接受当前周期、本角色、当前待检查响应且引用声明内的行动。角色收到的是自己的感知，不是完整世界状态；私人认知不进入他人感知或世界事实。

**Concordia 接入** 放在 `integrations/concordia/` 和必要的自定义角色组件中。领域消息、World 接口和响应屏障不导入框架（有测试守护该边界）。`integrations/concordia/tracing.py` 把领域追踪记录写入锁定版本的 `SimulationLog` 结构化日志，并为 Concordia 组件提供写入同一日志的 logging channel，使感知、响应、检查、打回与错误可沿周期/角色/响应/行动身份检索。`integrations/concordia/control.py` 把锁定版本的 `StepController` 适配为引擎运行门（运行、暂停、继续、单步、停止；许可在每个完整周期前的安全边界取得——初始化也是安全边界，首次事件处理前先等待许可；等待不占用事件循环，取消与停止释放等待线程；暂停不打断 World 提交、不推进剧情时钟），并把单周期报告转换为真实 `StepData` 广播给 `SimulationServer`——包含剧情毫秒、同刻全部事件（含行动节点类型与身份）、完整主动声明、登记后的行动记录、World 实际行动与被动过程的客观诊断、角色可见感知与自身行动状态、事实增量与结算结果、诊断变化原因；运行结束时以真实最终时刻与停止原因构造收尾步数据，不复播旧报告。观察者/诊断内容不进入角色感知。Concordia 的默认循环不能自动替代本项目的调度约定。

## 配置与统一运行入口

`config/loader.py` 提供严格文件配置加载（`load_assembly_config`；`load_run_config` 兼容核心文档的两字段秒配置）：JSON 所有层级必填/未知/重复键/类型/bool-as-number/nonfinite 严格校验（错误带路径/字段），场景从 `scene_path` 显式加载（相对路径基于配置文件目录，不依赖进程 cwd），参与角色必须覆盖场景 kind=角色对象且不重复/不存在（预检在模型/服务器/输出文件创建前），初始事件秒统一转整数毫秒。加载不创建任务/模型/网络/日志。

统一命令 `uv run --locked python -m sceneweave <config> [mode]`：

```bash
# scripted 模式：脚本假模型，不调用真实 API
uv run --locked python -m sceneweave examples/configs/run.json \
    --scripted examples/configs/scripted.json

# live 模式：外部 LanguageModel 工厂（用户注入，不自动选择供应商）
uv run --locked python -m sceneweave config.json \
    --live my_module:my_factory --live-options '{"model": "xxx"}'

# live 模式：内置 OpenAI 兼容 Chat Completions 工厂（DeepSeek 小场景已实测连通）
cp .env.example .env          # 从项目根目录运行；填入 DEEPSEEK_API_KEY，.env 已被忽略
uv run --locked python -m sceneweave examples/configs/run.json \
    --live sceneweave.integrations.concordia.chat_completions:create_model \
    --live-options '{"base_url": "https://api.deepseek.com", "model": "deepseek-flash", "api_key_env": "DEEPSEEK_API_KEY"}'

# 内置后端可与录制、UI 组合
uv run --locked python -m sceneweave examples/configs/run.json \
    --live sceneweave.integrations.concordia.chat_completions:create_model \
    --live-options '{"base_url": "https://api.deepseek.com", "model": "deepseek-flash", "api_key_env": "DEEPSEEK_API_KEY", "timeout_seconds": 120}' \
    --record runs/live-record --ui 8080

# 录制与重放
uv run --locked python -m sceneweave config.json --scripted script.json --record record_dir
uv run --locked python -m sceneweave config.json --replay record_dir

# UI 模式：真实 SimulationServer（默认 8080；传 0 由系统分配端口）
uv run --locked python -m sceneweave config.json --scripted script.json --ui 8080
uv run --locked python -m sceneweave config.json --scripted script.json --ui 0

# UI 同样支持录制和重放
uv run --locked python -m sceneweave config.json --scripted script.json --record record_dir --ui
uv run --locked python -m sceneweave config.json --replay record_dir --ui
```

`--live` 的工厂签名是 `(scope: str, options: dict) -> LanguageModel`（Concordia 2.4 锁定版本的 `LanguageModel` 抽象类，须继承并实现 `sample_text`/`sample_choice`），各作用域实例独立；不自动选择 API/模型、不静默 fallback 为假模型。外部工厂接口示意（用户注入的后端由用户自行验证）：

```python
# my_module.py — 接口示意，不是可复制代码
from concordia.language_model import language_model


class MyModel(language_model.LanguageModel):
    # 用户自行实现 sample_text/sample_choice，调用自己的模型后端
    ...


def my_factory(scope: str, options: dict) -> language_model.LanguageModel:
    return MyModel()
```

统一入口使用 Concordia 的 `LanguageModel` 抽象接口，具体模型后端由工厂注入。`--live` 在配置文件校验通过后、构造 live 工厂前，用 python-dotenv 显式加载当前工作目录 `.env`（路径固定为 `Path.cwd()/'.env'`，不向上搜索；`override=False` 保留终端已设置的变量；`interpolate=False` 不解析变量引用；缺失文件正常沿用环境变量）。`--scripted`、`--replay` 与 `--help` 不读取 `.env`；程序化调用工厂仍只使用调用方已传入的环境变量。加载异常按配置错误以非零状态退出，只报异常类别，不回显文件内容或任何值。

### 内置 Chat Completions 后端

`integrations/concordia/chat_completions.py` 的 `create_model` 提供项目内可直接引用的真实后端：用锁定版本的 [openai Python SDK](https://github.com/openai/openai-python)（`pyproject.toml` 固定 `openai>=2.54,<3`）调用 OpenAI 兼容的 `chat.completions.create`，协议编码、传输和响应反序列化由 SDK 完成，项目侧只做配置/采样参数校验、需要的明确终止判定和安全异常映射。

- 必填 `base_url`（仅 http/https，禁止用户名密码、query、fragment；非法主机/端口/控制字符或反斜杠明确拒绝，错误不回显用户名、密码、query 或原文）、`model`、`api_key_env`（环境变量名，不接受明文密钥）。可选 `timeout_seconds`、`max_tokens`、`temperature`：要么省略使用默认，要么给有效值，显式 JSON `null` 明确拒绝。供应商专用选项只能放进严格受控的 `extra_body`，且不得覆盖 `model`/`messages`/`stream`/`n`/`stop`/`seed`/`max_tokens`/`temperature`/`top_p`/`top_k`/`timeout`/认证或工具字段。未知、类型错误、bool 冒充数值、非有限值、空值与冲突都明确拒绝。
- 采样参数策略：`temperature` 与 `max_tokens` 取工厂配置或调用参数（两者同时显式给出且不一致时拒绝）；`top_p` 固定发送 Concordia 默认值 0.95；非默认的 `top_p`/`top_k`/`terminators` 和非 `None` 的 `seed` 目前不支持并显式拒绝。默认 `top_k` 仅用于接口兼容，不向 Chat Completions 透传。
- 每个作用域一个独立实例；每次 `sample_text` 新建并关闭独立连接（不引入长期 HTTP 连接），`max_retries=0`、禁止重定向（不把密钥转发到其它 host）；请求只含单条 user 消息、`stream=false`、`n=1`，不共享对话历史。
- 客户端关闭失败明确报错；若请求已有错误，关闭失败只追加不含原始正文的说明，保留请求错误。
- 只接受 `finish_reason="stop"`、`message.role="assistant"` 且 content 为非空字符串的单个文本响应；截断（length）、工具调用、空/null、choice 数量异常、错误角色、缺字段、非 JSON 都抛明确异常，错误消息只报稳定类别与数量，不插入上游字段值；坏输出不会被当作成功文本。超时、连接失败、401/429/500 等分类为可诊断异常并携带状态码，异常消息不含密钥、Authorization、上游正文或提示原文。
- 只从 `api_key_env` 指定的环境变量读取密钥，不读 `.env`、不使用 `OPENAI_API_KEY` 等全局回退；统一 CLI 的 `--live` 会先把当前工作目录 `.env` 显式加载进进程环境（见「配置与统一运行入口」），适配器自身仍只读已传入的环境变量。SDK 的组织/项目/管理员/自定义请求头环境变量被显式压制（检测到 `OPENAI_CUSTOM_HEADERS` 直接拒绝）。
- `timeout_seconds` 是 SDK/HTTP 客户端的连接与读写等待超时，不是整次调用的绝对墙钟 deadline；本项目没有整次调用的墙钟总预算。
- `sample_choice` 不在本项目运行链中，显式 `NotImplementedError`，不伪造概率或默认选项。

`--replay` 只读录制目录，不导入 live 工厂、不读密钥、不联网。示例供应商（DeepSeek）的请求/响应约定参考[官方 Chat Completion 文档](https://api-docs.deepseek.com/api/create-chat-completion)；供应商与模型必须由用户显式配置，适配器不自动选择，也不把 SDK 默认当作实际供应商。2026-10-06 已用 `deepseek-flash` 跑通仓库小场景：9 次真实调用、2 个完整周期，在 8000ms 剧情时限正常停止，无运行错误；角色选择维持现状，没有新增行动。这只验证当前小场景的真实调用链，不代表复杂互动效果、提示调优、速度或费用已验证。

作用域命名空间（避免角色 ID 与 World 作用域碰撞）：`role:{角色}`、`translator:{角色}`、`world:check:{角色}`、`world:progress`、`world:adjudicate`，每作用域采样器只构造一次并缓存，录制目录按作用域独立文件。角色私人 background 只进该角色的 `ModelRole` 提示，不进入 World 场景或其他任何作用域的提示（有测试断言零泄漏）。

每次运行输出目录独立（默认 `sceneweave-runs/run-{id}`，可用 `--output` 指定；已有目录不覆盖；录制目录与输出目录不得相同或嵌套），包含 JSON 结构化日志、运行概要（停止原因/最终时刻/完成周期数/世界版本），录制运行额外写入 manifest（绑定实际场景事实、角色私人背景、初始事件、运行参数、运行身份与全部作用域；成功及失败运行均保存）。运行错误以非零退出并保存引擎错误快照；成功运行若导出 IO 失败同样非零退出（链式保留 IO 原因）；取消/异常的概要从引擎公开状态读取真实时刻与已完成周期，不伪造零值。`--record` 与 `--replay` 互斥；重放只使用录制目录，不导入 live 工厂，成功导出前校验录制条目全部消费，剩余或请求错配明确失败。

真实模型采样（live）与重放回放的组件日志写入同一 `SimulationLog`：每次采样以 `entry_type=component` 记录 Prompt/Response 原文与作用域、周期、剧情时刻身份（step 取自周期索引，不以墙钟冒充剧情时间）；重放不建真实模型，但每次回放的服务内容同样可查询。采样器错误日志保留原始异常，错误追踪失败作为附加说明；组件通道由运行编排注入（采样器公开 `bind_channel_for`），采样器自身不创建日志。

### 秘密文件争夺场景

`examples/configs/run-contest.json` 与 `scene-contest.json` 提供两人争夺同一份密封秘密文件的独立示例。林舟与沈言分别受命亲自取回文件，都希望自己保管；各自知道文件在书桌上，但不知道文件内容或对方的私有指令。两人都已在书桌前、可以伸手触及文件；角色自行决定取走、交涉、阻止或退让，不预设胜者，也不预设谁已取到文件。

0 秒开场事件打开书房照明，并解除文件的暂时禁取状态。World 裁决实际状态变化，经校验、提交后按可见范围投递感知；竞争动机则通过各角色私有 `background` 进入 System1/2/3。配置不会强制 System1 重新考虑，也不会直接登记角色行动。同刻争夺沿用 World 联合裁决与对象树的单一容纳关系。

模型提示保持既有 schema 与严格校验：System1 输出三路分流 `continue/replan/rethink`；System3 输出一条一维全量结构化原子动作链（含尚未满足依赖的后续动作），不出现 maintain/modify/new_actions 分类或 feasible/retry 包装；System2 只产当前意向（无 `abandon`）。World 检查在整链翻译之后进行，输入携带完整链与完整排程以及本角色引擎侧已登记安排；等待中的维持不因尚无实际行动状态被拒，安排只供检查、不进入感知或事实；检查提示按完整链列出 `action_ids` 白名单，不通过时给具体受影响行动或空数组；检查不因前置未执行就一概拒绝合理链。翻译器只设置必要过程节点，且节点严格位于开始与结束之间。

结算结果与节点类型的允许规则归口在 `domain/adjudication.py`，引擎提交前校验与 World 提示共用同一映射。仅修改或撤回时不得标成 failed/completed；同身份兼有其它节点时以合并的允许集合为准。撤回可不提供 outcome，环境事件只写事实变化，`follow_ups` 只登记独立环境过程，不重复安排已登记行动的节点。`passives` 有两种形态：纯描述只给 `id` 与 `description`；确实存在被动物理运动时才给 `object_id`/`speed`/`direction`，且三项必须成组同时出现，部分给法在解析时明确报错、不静默忽略。对象归属未确认只通过 `outcomes` 与事实表达，不虚构物理被动过程；`stop_movements` 的引用必须来自本轮参与行动或确有进行中运动的行动，本轮 outcome 为 unconfirmed 的行动不得列入，未确认结果不得隐式停止已知运动。`passives` 只登记本轮新出现的被动影响，已有未结束的持续保留、重复声明会被拒绝，进展在 `passive_updates` 中维护、结束用 `stop_passives` 引用已存在身份；主动持握/守护属于主动行动，动作到 end 也不得把仍在主动维持的行为改写成被动。联合裁决提示另明确每个 start/checkpoint/end 节点都必须给出 outcome；进展提示明确 `updated_action_ids`/`action_progress` 只引用进行中行动。

真实 DeepSeek Flash 的完整 12 秒运行已验证通过：按时限正常停止，完成 12 个事件周期，最终 World 版本为 15，无运行错误。林舟在 1.8 秒完成取文件，至时限仍持有唯一一份密封文件；沈言交涉未能取得文件，随后提交报告请求上级裁定，裁定结果尚未返回。这是时限停止，未定义场景自然结束或预设胜者。

录制离线回放与原概要完全一致，文件对象与父归属唯一；真实 Chromium 已验证网页全部周期、角色/阶段筛选、完整模型输入输出及刷新恢复。对应本机记录为 `sceneweave-runs/contest-secret-20261006-v13-record`（运行数据不进 Git）。下方命令与本轮验证一致，按 [DeepSeek 思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode/) 使用低强度思考、16000 tokens 输出预算与 JSON 输出模式；目录名须未被使用，实际行为以各次记录为准。

从项目根目录运行，使用本机 `.env` 中已配置的 `DEEPSEEK_API_KEY`：

```bash
uv run --locked python -m sceneweave examples/configs/run-contest.json \
    --live sceneweave.integrations.concordia.chat_completions:create_model \
    --live-options '{"base_url": "https://api.deepseek.com", "model": "deepseek-flash", "api_key_env": "DEEPSEEK_API_KEY", "timeout_seconds": 120, "max_tokens": 16000, "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "low", "response_format": {"type": "json_object"}}}' \
    --record sceneweave-runs/contest-record \
    --output sceneweave-runs/contest-output \
    --ui 8080
```

打开页面后点击“运行”；录制和输出目录须尚不存在，再次运行时换目录名。内思考间隔 5 秒，最长剧情时长 12 秒；达到时限会保留未完成行动，不代表争夺已自然结束。

原 `run.json` 与 `scene.json` 保留。重放会严格匹配提示原文，旧录制须使用其对应版本的代码；本次阶段提示调整后，请用新录制进行当前版本的回放。它们的开场只有静态环境描述，真实运行中没有提交事实变化；角色背景也只有人格，没有当前目标，因此该次两名角色都选择等待。

### 运行后审计（观察者，不回写模拟）

`--audit-config PATH` 在模拟结束并导出原日志与 `summary.json` 后，先用固定证据报告做确定性机械检查，再由配置的模型独立审计同一报告。缺省不开启审计、不读 `.env`；显式开启后（包括 `--replay`/`--scripted`）按既有 `override=False`/`interpolate=False` 规则加载当前目录 `.env` 供审计模型读取环境变量。审计不进入角色感知、不改 World/引擎、不修改录制或已导出事实；模拟 `finished` 仅表示模拟结束。

审计配置严格校验：根对象只允许 `models`；`models` 非空，每项含唯一 `id`、可选 `factory`（`module:callable`，默认内置 `chat_completions:create_model`）与 `options`；`options` 不得出现明文密钥字段，只传 `api_key_env`。默认示例 `examples/configs/audit.json` 用已验证接入方式的 DeepSeek Flash（`extra_body` 含 thinking enabled、`reasoning_effort=low`、`response_format=json_object`）；`examples/configs/audit-multi.json` 是双模型模板，第二项为占位 endpoint/模型名/环境变量，需用户填入真实 OpenAI 兼容 endpoint（例如其他兼容供应商）后再运行（不声称任何未验证供应商已可用或已调用）。当前范围是 OpenAI 兼容 endpoint 上的多模型审计。

输出写入同一次运行的输出目录：`report.md`（场景、规则投影、逐周期追踪与机械检查）、每模型 `audit-<id>.prompt.txt`/`audit-<id>.response.txt`、`audit.json`、`audit.md`；不打印或保存密钥与工厂选项。报告是投影：原始组件调用提示与重复指令不整段复制（完整原文可在 `simulation_log.json` 按 stage/周期/角色检索），但保留场景事实、summary、完整周期边界、角色实际感知与声明、翻译安排、World 候选完整结构输出、校验/提交/撤回/进展等失败与关键证据；模型原始回答与候选明确标为非已提交事实。机械检查只根据明确结构报确定问题，缺足够追踪标“未覆盖”而非通过，并对通信正文、语义先后、同刻批次等未覆盖范围给出明确说明；模型输出按固定 schema 严格校验，findings 的 `evidence_refs` 必须至少引用一条报告证据且不得虚构，非法输出只报稳定类别与字段位置、不转述模型原文。一个模型失败不阻止其他模型，也不改写模拟概要、世界事实或录制；审计失败或取消时 CLI 非零退出/明确提示但仍保留成功的模拟概要。

```bash
uv run --locked python -m sceneweave examples/configs/run.json \
    --scripted examples/configs/scripted.json \
    --audit-config examples/configs/audit.json \
    --output sceneweave-runs/audit-demo
```

上面的 `run.json` + `scripted.json` 是仓库内实际配套的离线模拟；`--audit-config` 会调用真实审计模型，需要 `.env` 中的 `DEEPSEEK_API_KEY`。

UI 模式同样支持 `--audit-config`：页面新增独立“运行后审计”区，显示等待模拟结束/机械检查/模型运行/完成/某模型失败等真实状态，多个模型结论并列；模拟结束后仍会轮询到审计更新，Markdown 导出附独立审计章节。

## UI 运行

统一入口 `--ui [PORT]` 启动真实 Concordia `SimulationServer` + `StepControllerGate`：初始暂停在首次事件前（无模型调用），单步完成一个完整事件周期，play 连续运行、pause/stop 如实生效；每个已完成周期广播一次真实 `StepData`（同刻事件、各角色完整声明与可见感知、World 事实与结算、诊断），运行结束广播真实最终状态（最终时刻/停止原因/错误/世界版本），页面保留至回车/EOF/Ctrl+C 后关闭服务器与线程。端口 0 时实际地址在启动后打印。合法角色身份若与界面保留键冲突，则使用唯一呈现别名并保留原始 `role_id`，避免覆盖 World 事实或诊断。

页面（`app/page.py`）不直接拼接 SSE 帧，而是通过固定只读端点 `GET /run-snapshot` 读取完整本次运行数据；SSE 与 1.5 秒轮询只用于触发重新拉取。因此实时运行、刷新页面、SSE 重连和运行结束后首次打开看到的是同一份完整历史，不会重复周期或丢失历史：

- **运行过程**：按时间顺序保留每个已完成事件周期的真实 `StepData`——同刻全部事件（含行动节点身份与内容）、各角色完整主动声明、可见感知与自身行动状态、统一登记后的行动记录、World 实际行动与被动过程的客观诊断、事实增量与结算结果、行动变化原因。最新周期默认展开，其余可折叠。页面采用居中浅色阅读布局：周期作为时间线，摘要显示剧情时刻与事件/角色/登记数量及各结算结果计数（混合周期不整体着色成功，未确认用单独的“结果未知”样式）；周期内角色声明与感知以角色卡片分组，登记/实际状态/被动过程/事实结算分区呈现，长技术块默认折叠但完整保留（超过 5 条的感知也在展开区完整保留）。
- **模型调用**：同一 `SimulationLog` 中的真实组件调用（System1/2/3、翻译器、World 检查/进展/裁决）以及跳过与阶段错误记录，显示作用域/角色、阶段、周期身份与剧情毫秒，可按角色/作用域与阶段筛选；默认显示从真实回答字段提炼的短摘要（System1 的 `route`/`note`、System2 的 `intention`、World 检查的 `passed`/`reason`、System3/翻译器的安排计数、进展/裁决计数等），缺字段、非 JSON 或调用失败均有如实说明，不伪造发生过的内容。点击展开查看完整 Prompt/Response/Error 原文；内容引用在服务端解开，浏览器不会看到 `_ref` 哈希。
- **最终收尾状态**：真实停止原因、完成周期数、最终剧情时刻与错误，与周期历史分开呈现；完成信号到达后控制按钮停用，陈旧 `/status` 响应或晚到的旧快照不会覆盖完成态。
- **视图与导出**：顶部可切换「全部过程 / 成功动作转录（只看已完成）」，并在运行中/结束后随时「导出 Markdown」。成功动作转录只把已提交周期中 World 结算 `result=completed` 的 `detail` 原文按剧情钟 `mm:ss[.毫秒]`（毫秒去尾零，分钟不限两位）排成按时序的动作正文，默认不显示周期/行动身份与状态码；每行可展开「查看结算详情」查看结果、行动身份、剧情时间与周期。候选输出、检查通过、普通模型调用与失败/未确认/撤回结算不进正文，混合周期不整体标为成功，顶部全部周期总数与真实剧情时刻不变；该模式下模型调用区隐藏，最终收尾状态与运行后审计改为可折叠（后台仍持续更新，不被误标为成功动作），全部模式保持完整展开。切换与轮询不修改快照、日志或引擎状态并保持所选模式。Markdown 在浏览器本地以 UTF-8 `.md` 下载，跟随当前视图：成功动作转录导出同样 `[mm:ss] 动作正文` 顺序，正文只含成功动作，最终状态与审计作为分隔附录；全部过程导出周期摘要与当前角色/阶段筛选后的调用摘要（省略原始 Prompt/Response）并附审计。不访问新端点、不导出凭据，没有最终状态或尚未开始时也有有效说明。

页面所有动态文本（模型回答、角色声明、事件描述等）都经 `textContent`/安全 DOM 构建，恶意内容不会被当作 HTML 或脚本执行。快照只在服务器进程内存中存在，服务器关闭即消失（CLI 会在运行结束后等待回车期间保持可查看）；需要长期离线查看时使用输出目录的 `simulation_log.json`。快照端点不接受参数、路径或查询条件，只读同一 `SimulationLog` 与已广播历史，不读取环境变量、密钥或磁盘文件。

`app/ui_snapshot.py` 的 `SnapshotSimulationServer` 是 `SimulationServer` 的轻量子类：继承全部原有控制/状态/SSE 行为，仅捕获已广播步数据并新增该固定端点；不修改 site-packages，也不改变调度与模型流程。

可选的真实浏览器测试 `tests/test_ui_browser.py` 需要本机 Node、Chromium，以及环境变量 `SCENEWEAVE_TEST_PLAYWRIGHT_CORE` 指向已安装的 `playwright-core` 目录；依赖缺失时明确跳过，其余 HTTP/历史/日志测试照常运行。

## 演示运行

不调用真实模型、不读取密钥，使用脚本角色、假语义世界与脚本翻译器：

```bash
# 离线演示：动作持续、感知、检查打回、执行中撤回与被动影响、时限停止
uv run --locked python examples/offline_demo.py

# 本地 UI 演示：Concordia SimulationServer + 薄 HTML 视图（默认 8080 端口）
uv run --locked python examples/ui_demo.py [端口]
```

两个演示都通过 `SimulationLogTracer` 把引擎/世界/角色/翻译的因果链写入 Concordia 结构化日志（含翻译客观上下文、既有安排、裁决节点与修改内容等关键输入/输出），运行结束导出 JSON 文件供离线检索。[模型翻译器演示](examples/model_translator_demo.py) 展示 ModelRole + ModelTranslator + Scheduler + 结构化日志的生产装配形态（脚本假模型，可替换为 `LanguageModelSampler` 真实桥）：`uv run --locked python examples/model_translator_demo.py`。

[模型 World 演示](examples/model_world_demo.py) 展示场景卡加载 + ModelWorld + ModelRole + ModelTranslator + Scheduler 的完整生产链路（内联场景数据、脚本假模型、日志导出）：`uv run --locked python examples/model_world_demo.py`。

[System1 路由探针](docs/System1路由探针.md) 是独立于真实模拟的实验脚本，只调用 System1 比较 `route` 字段与三路倾向分布两种表达（不启动 System2/3/翻译器/World/引擎，不影响正式路由）：`uv run --locked python scripts/probe_system1_routing.py --dry-run`（真实调用见文档，默认忽略目录输出）。第二轮敏感度探针（event intensity sweep + 真实角色/无人格基线的 profile 差异）见同文档：`uv run --locked python scripts/probe_system1_sensitivity.py --dry-run`。

示例 `ui_demo.py` 使用自带的精简页面（只显示最新步数据），与统一入口 `--ui` 的完整历史/模型调用视图相互独立；两者都由真实 `StepController` 驱动运行、暂停、单步与停止，回车退出或 Ctrl+C 中止都会清理服务器与控制器。`SimulationServer` 按框架默认绑定全部网络接口，请在可信环境运行，不要公开部署。

## 当前实现范围

已按[模块交互契约](docs/模块交互契约.md)实现跨模块消息与身份：处理周期、角色、响应、行动、提交组与共同世界状态版本均有明确身份；消息为不可变结构（嵌套集合一律元组、映射冻结为只读视图），构造时校验非空身份、字段类型、行动归属、重复身份与关联一致性，bool 不会被当作整数时间。剧情时间为非负整数毫秒，秒配置经 `domain/time.py` 的入口转换函数转为整毫秒，无法整毫秒表示、负数与非有限值直接拒绝；文件配置加载见「配置与统一运行入口」。

`Role` 抽象类实现 `async perceive(TurnInput)`、`async rejected(Rejection)`，以及整链检查通过后的 `save_plan`（角色内部原子保存完整链与全部排程）与 `execution_submission`（最近步骤执行提交）契约。`RoleResponse` 携带周期/角色/响应身份、可选的完整一维动作链 `plan` 与本轮要声明的 `actions`：结构化活跃路径用 `StepAction`（原子动作 + 真实身份 + 已解析依赖），旧三类声明（维持/修改/新增）仅为低层历史测试保留。合法空集合表示撤回全部主动维持。

`World` 接口提供无副作用的 `check_response` 检查入口（接收本角色只读既有安排快照，仅为语义检查提供已登记等待也可维持的上下文）、按流逝剧情时间更新进展的 `update_progress`（实际变化原子应用并递增共同状态版本）、供翻译器使用的只读 `translation_context` 与对外只读 `diagnostic_snapshot`（实际行动状态与被动过程），并把裁决候选（基于本时刻事件提交前、World 已更新相关行动进展后的共同状态；事件与行动节点携带完整内容、执行角色归属与登记安排，修改裁决携带明确变化与新旧未来安排）、运行时校验（结果关联候选与裁决前版本）、实际提交（含结构化行动结算结果）、发布拆成明确边界。`ModelWorld`（`world/model_world.py`）是生产可调用的模型驱动实现：不可变场景对象树（`world/scene.py`）承载事实，结构化补丁（set_state/move/set_position）经候选校验后原子应用；检查/进展/联合裁决按作用域采样器调用模型，严格 JSON schema（全部数组字段必填、未知/重复字段与非有限数值拒绝、open/locked 只接受布尔、unit 仅米、运行时补丁房间边界校验）；候选以对象身份（is）核对生成现场——值相等伪造、重复提交、过期版本/基线、模型调用在途全部拒绝；commit 在本地副本构造全部新状态后一次性切换（失败零局部事实，合法无变化事件允许提交并严格递增版本）；publish 是同步确定性投影——感知由已提交结构与提交后快照生成（不广播模型自由描述，隐藏内容不进入任何角色输入，打开容器传递新可见内容、关闭不宣告藏品）；检查/写入互斥（检查期间版本冻结并报告进入时版本），检查留痕携带经公开进展流转的真实剧情时刻（失败/取消不切换当前时刻）；进展四数组字段必填，覆盖持续行动与被动过程（显式 done 才结束、省略不等于删除；被动运动元数据速度/方向/归属随 commit 批量保存与清理、完整参数进进展提示、位置由进展模型统一提案不做双重积分），无实际变化不递增版本；运动归属参与行动执行角色、方向归一化、终态后不得重启、未确认不隐式停止、冲突补丁整批拒绝。感知使用统一受众过滤 helper：关闭容器硬约束优先（注入策略返回 True 不能看见关闭容器后代），策略只能收紧；检查输入、发布目标/新容纳位置/打开后新可见内容全部经同一 helper 按角色过滤；打开容器只枚举该角色确实可见的后代（关闭的子容器本身可见但其后代仍隐藏）；感知物理默认只是关闭容器过滤，未做精确视线/寻路/碰撞。`ResponseRunner` 一次采集场景全部角色的完整声明：输入校验、角色并行/同角色串行、响应身份当前周期内全局唯一（含跨角色，`CycleOutcome.used_response_ids` 供后段重试延续检查）、全部返回后核对行动身份不跨角色重复；它不做 World 检查。整链翻译、World 整链检查、角色保存与最近步骤执行提交在屏障之后由 `Scheduler` 统一编排，共同状态版本在本轮固定并在结束时复核。

`Scheduler` 实现事件调度与行动生命周期闭环：整数毫秒剧情时钟按最近事件跳转，同刻整批联合裁决；进入时刻后先更新进展再裁决，事件事实提交后才生成角色输入；全部角色通过屏障后翻译与登记校验（结构不一致是运行错误，时间约束经 `rejected` 同周期重试），统一登记先完整准备登记计划与队列候选再提交，修改与撤回的客观效果经 World 结构化裁决后统一生效；维持不重排也不重新翻译，遗漏撤回主动维持（未开始撤销未来事件、执行中由 World 裁决客观效果并保留被动过程），修改只调整未来节点且在 World 实际生效，终态身份不可复用，未确认不是已知终止；行动完成/失败只随 World 提交的结算结果变化，结算结果与节点类型的兼容性在事实提交前预检。周期性内思考按 `RunConfig`（`engine/config.py`，秒入口转换、正数校验）从起始时刻触发并与同刻事件合并；运行时限包含边界、越界事件保留，跳到无事件的时限边界时先按流逝剧情时间更新实际进展；自然结束（注入判断，空响应不算）、时限、用户中止与运行错误分别明确，运行错误的实例不可再次推进，错误现场可导出 JSON 快照（周期、已取出批、队列、行动记录、阶段、世界版本、候选与未生效计划）。`CycleReport`（`engine/report.py`）区分角色可见视图、登记后行动记录与 World 客观诊断，供界面转换使用。

Concordia 接入：结构化/组件日志见 `integrations/concordia/tracing.py`；运行控制与界面步数据见 `integrations/concordia/control.py`——真实 `StepController` 适配为运行门（暂停不打断 World 提交、不推进剧情时钟；本项目没有墙钟总预算，仅模型请求自身可有超时），真实 `StepData` 广播到 `SimulationServer`（`tests/test_concordia_control.py` 以真实服务器与 HTTP 端点验证）。可运行演示见「演示运行」一节。

模型驱动的 System1/2/3 已由 `ModelRole` 与 `roles/components/` 落实：组件经可注入的 `TextSampler` 协议调用模型，`integrations/concordia/sampling.py` 提供真实 Concordia `LanguageModel` 的异步桥（采样经真实 `ComponentWithLogging` 组件把原始输入/输出写入 `SimulationLog`；`channel_for` 按阶段建立通道，datum 标记实际阶段），以及按作用域（角色）隔离的录制/重放：跨角色并行采样的完成顺序不确定，共享单个录制文件无法重放，`RecordingDirectory`/`ReplayDirectory` 为每个角色生成独立文件即可确定性重放整个多角色运行；录制包含失败调用（重放重现原错误身份），请求逐条严格匹配、错配不消费记录。注意：线程中的同步模型调用无法被硬取消——取消/超时只放弃等待，底层调用会继续到自然返回。阶段记录（提示、原始输出、解析结果、跳过/重试/错误与周期/角色/响应/行动/剧情时间身份）写入与调度器相同的因果链，追踪器自身失败不掩盖模型/解析异常。每个阶段（含跳过）都有测试与端到端集成（ModelRole + 假 World/Translator + Scheduler + 结构化日志 + StepData，含双角色并行录制重放回归）。生产语义的具体提示工程与真实模型参数调优未在本批验证（测试与演示使用脚本假模型）。

尚未实现：场景自然结束条件的场景侧定义（`should_end` 可由装配注入，具体场景规则需自行提供，时限不是自然结束）；内置 Chat Completions 工厂已提供，并以真实 openai SDK 连本地假 HTTP 端点覆盖协议参数、错误/超时/重定向、组件日志与录制重放；DeepSeek 真实小场景已验证连通，复杂互动效果、提示工程与模型参数调优仍待验证（不声称速度或费用）。统一入口 `--ui` 已接入真实 `SimulationServer`/`StepControllerGate`/周期与最终广播，并新增完整历史与模型调用视图（`app/ui_snapshot.py`、`app/page.py`，`tests/test_ui_snapshot.py`、`tests/test_ui_browser.py` 覆盖）；组件日志通道由运行编排统一注入（含录制/重放模式）。其余缺口以可注入接口承接，示例中的脚本实现明确标注为演示用假件（脚本世界裁决的内置逻辑仅为演示行走语义，不是通用 World 物理；显式提供的 World 组件脚本优先执行，耗尽直接报错）。
