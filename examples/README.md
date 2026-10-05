# 示例

场景组装与演示放在这里：配置 World、角色和引擎，并注入模型客户端。
核心模块不导入示例。当前已实现事件调度与行动生命周期的运行闭环
（见 [README](../README.md)），下面的演示可以实际运行；模型驱动的
System1/2/3、翻译器与 World 语义仍以注入接口承接，示例中的语义依赖
都是脚本假件。

- [离线演示](offline_demo.py)：脚本角色 + 假语义世界 + 脚本翻译器，
  实际运行动作持续（位置按剧情时间推进）、感知投递、检查打回、
  执行中撤回与被动影响、运行时限停止：

  ```bash
  uv run --locked python examples/offline_demo.py
  ```

- [本地 UI 演示](ui_demo.py)：Concordia `SimulationServer` + 薄 HTML
  视图，SSE 接收真实 `StepData`（剧情毫秒、同刻事件、主动声明、登记
  记录、World 实际行动与被动诊断），按钮驱动真实 `StepController`
  的运行/暂停/单步/停止；按回车退出并清理服务器：

  ```bash
  uv run --locked python examples/ui_demo.py [端口]
  ```
