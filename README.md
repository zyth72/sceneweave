# sceneweave

使用 uv 管理、基于 Google DeepMind Concordia 的角色模拟项目。

- [核心循环与角色职责说明](核心循环与角色职责说明.md)
- [代码规范](代码规范.md)
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
├── domain/messages.py           # 事件、感知、行动提议与角色响应
├── world/base.py                # World 接口：事实提交、结算、感知生成
├── engine/responses.py          # 引擎中的本轮并行响应屏障
├── roles/base.py                # 角色统一 respond 入口
├── roles/components/            # 角色内部 System1/2/3 组件
└── integrations/concordia/entity.py  # 框架实体接入
tests/                          # 不依赖真实模型的测试
examples/                       # 场景组装与演示
```

**World** 保存客观事实。事件到达时通过 `apply_event` 落地；行动到达结算时刻时通过 `settle_action` 判断实际结果。只有事实提交完成后，World 才返回按角色感知范围生成的 `Delivery`。角色不能拿到 World 的可变状态引用。

**完整引擎** 负责保存剧情时钟、事件队列及已登记的行动，调用 World 并将其投递交给角色。当前已实现 `ResponseRunner`：每个被通知的角色通过自己的 `respond` 处理本轮输入；所有响应完成后，整体返回经过身份检查的行动提议。固定的返回顺序只用于稳定呈现，不代表同刻裁决规则。

**角色** 保存自身认知、意向和内部状态。`Role.respond` 是对外接口；System1/2/3 的协作在角色内部实现，相关组件放入 `roles/components/`。角色收到的是自己的感知，不是完整世界状态。

**Concordia 接入** 放在 `integrations/concordia/` 和必要的自定义角色组件中。领域消息、World 接口和响应屏障不导入框架。Concordia 的默认循环不能自动替代本项目的调度约定。

## 当前实现范围

已建立可安装的 uv 包、共享消息、World/角色抽象接口及并行响应屏障。响应屏障不修改剧情时间或登记行动，调用失败时不返回部分行动；角色内部状态可能已更新，错误恢复不能直接重放。

事件队列、剧情时钟、行动生命周期、具体世界模型和 System1/2/3 尚未实现。Concordia 接入也不代表已经实现三层推理。同刻处理顺序、冲突裁决和结束规则须先明确设计，再实现完整引擎。
