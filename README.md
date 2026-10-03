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
│   ├── translation.py            # 行动翻译输入输出消息
│   ├── adjudication.py           # World 裁决候选与校验提交结果
│   └── tracing.py                # 领域追踪接口（不依赖 Concordia）
├── world/base.py                 # World 接口：检查、裁决候选、校验、提交与发布
├── engine/
│   ├── responses.py              # 一次处理周期的全角色响应与检查/打回屏障
│   └── registration.py           # 登记前时间安排纯校验
├── roles/base.py                 # 角色 perceive/rejected 交互契约
├── roles/components/             # 角色内部 System1/2/3 组件（尚未实现）
└── integrations/concordia/
    ├── entity.py                 # 框架实体接入
    └── tracing.py                # 领域追踪接入 Concordia 结构化/组件日志
tests/                          # 不依赖真实模型的测试
examples/                       # 场景组装与演示
```

**World** 保存客观事实，并提供对完整主动行为声明的无副作用检查入口 `check_response`（只产生记录共同状态版本的 `ActionCheckResult`，不调用角色、不提交事实、不取消行动）。事件与行动的落地拆成明确边界：`adjudicate` 产生联合裁决候选，`validate_adjudication` 做运行时校验，`commit_adjudication` 原子提交实际状态变化，`publish` 才生成按角色范围过滤的 `Delivery`。具体事实模型与语义判断由实现注入，角色不能拿到 World 的可变状态引用。

**引擎** 负责保存剧情时钟、事件队列及已登记的行动（尚未实现）。当前已实现 `ResponseRunner`：一次处理场景全部角色的输入与响应——无外部感知的角色也有明确输入，同刻事件合并；角色并行处理、同角色串行，检查不通过经 `rejected` 打回并重新检查（有明确上限），通过后保存候选，全部角色完成后整体返回 `CycleOutcome`，保留响应与提交组边界、不扁平化。屏障自身不拥有或推进剧情时钟，也不登记、撤回或提交客观事实；共同状态前提在周期开始时固定。`engine/registration.py` 提供登记前时间安排的纯校验。固定的返回顺序只用于稳定呈现，不代表同刻裁决规则。

**角色** 保存自身认知、意向和内部状态。`Role` 抽象类实现 `perceive(TurnInput)` 与 `rejected(Rejection)` 交互契约：前者接收本轮输入并返回完整主动行为声明，后者接收 World 的打回反馈并返回新的完整声明。System1/2/3 的协作在角色内部实现，相关组件放入 `roles/components/`（尚未实现）。角色收到的是自己的感知，不是完整世界状态。

**Concordia 接入** 放在 `integrations/concordia/` 和必要的自定义角色组件中。领域消息、World 接口和响应屏障不导入框架（有测试守护该边界）。`integrations/concordia/tracing.py` 把领域追踪记录写入锁定版本的 `SimulationLog` 结构化日志，并为 Concordia 组件提供写入同一日志的 logging channel，使感知、响应、检查、打回与错误可沿周期/角色/响应/行动身份检索。Concordia 的默认循环不能自动替代本项目的调度约定。

## 当前实现范围

已按[模块交互契约](docs/模块交互契约.md)实现跨模块消息与身份：处理周期、角色、响应、行动、提交组与共同世界状态版本均有明确身份；消息为不可变结构（嵌套集合一律元组、映射冻结为只读视图），构造时校验非空身份、字段类型、行动归属、重复身份与关联一致性，bool 不会被当作整数时间。剧情时间为非负整数毫秒，秒配置经 `domain/time.py` 的入口转换函数转为整毫秒，无法整毫秒表示、负数与非有限值直接拒绝；尚未实现配置加载器。

`Role` 抽象类已迁移为 `async perceive(TurnInput)` 与 `async rejected(Rejection)` 契约，旧 `respond` 已移除。`RoleResponse` 携带周期/角色/响应身份与一份必须显式提供的完整主动行为集合（省略 `actions` 字段是构造错误，不会被当作空集合），其中维持声明、明确修改与本次新行动并行组三类在类型上可区分；合法空集合表示撤回全部主动维持。维持“保留原进展”、修改“只调整未来部分”当前只是类型语义，实际的进展保持与未来安排调整属于尚未实现的行动生命周期。

`World` 接口提供无副作用的 `check_response` 检查入口，并把裁决候选（基于本时刻事件提交前、World 已更新相关行动进展后的共同状态）、运行时校验（结果关联候选与世界状态版本）、实际提交、发布拆成明确边界；具体世界事实模型与语义判断由实现注入，当前只有测试用假世界实现。`ResponseRunner` 一次处理场景全部角色：输入校验（全部角色、输入键与角色一致、同一周期与时刻）、角色并行/同角色串行、检查与打回循环（明确上限，超限抛运行错误）、响应身份当前周期内全局唯一（含跨角色）、全部通过后核对行动身份不跨角色重复、共同状态版本在本轮固定并在结束时复核；异常不返回部分结果，也不承诺回滚角色私有状态。`engine/registration.py` 提供登记前时间安排纯校验，当前只覆盖本次新行动（维持无需重排；修改声明的未来安排尚未实现）：开始晚于当前时刻、结束晚于开始、过程节点合法、组内 max(start) < min(end)、引用/提交组/完整性；失败不拆组、不部分接受。Concordia 日志接入见上文；未实现运行控制与界面。

尚未实现：事件队列、剧情时钟推进与调度器、行动翻译器（仅交付消息类型与登记校验）、行动生命周期状态机、具体世界事实模型与同刻联合裁决编排、System1/2/3 组件、周期性内思考调度、运行时限与配置加载器。Concordia 接入也不代表已经实现三层推理；运行控制接口（`StepController`/`StepData`）、可视化界面与真实界面步数据尚未接入，完整引擎 StepData 不在当前交付内。行动开始时刻、同刻事件联合判断及运行参数的设计约定见[核心循环与角色职责说明](docs/核心循环与角色职责说明.md)，后续完整引擎须遵循这些约定。

Concordia 组件日志已通过结构化日志接入（见 `integrations/concordia/tracing.py` 及其测试）；运行控制接口与可视化界面，以及本项目剧情时间、行动持续性和结算语义的框架适配尚未完成，属于后续相关功能的强制交付要求，详见[开发规范](docs/开发规范.md#concordia-接入调试与运行语义强制)。
