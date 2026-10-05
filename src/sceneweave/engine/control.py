"""引擎的运行控制表面：可注入的运行门与运行状态。

运行门在完整事件周期的安全边界等待许可（暂停/继续/单步/停止），
不参与剧情语义。框架适配（如 Concordia StepController）在接入层
实现本协议；引擎只依赖此处的抽象。
"""

from dataclasses import dataclass
from typing import Protocol

from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.engine.report import CycleReport


class RunGate(Protocol):
    """运行许可门；等待不得推进剧情时间，也不计入墙钟超时预算。"""

    def should_stop(self) -> bool:
        """是否已请求停止；停止与正常结束是不同的停止原因。"""
        ...

    async def wait_for_permission(self) -> bool:
        """在周期边界等待运行许可；返回 False 表示请求停止。

        实现不得长期占用事件循环；被取消时应释放底层等待资源。
        """
        ...


@dataclass(frozen=True, slots=True)
class RunStatus:
    """注入的自然结束判断所能看到的运行状态快照。"""

    simulation_time: SimulationTime
    completed_cycles: int
    last_report: CycleReport | None

    def __post_init__(self) -> None:
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if isinstance(self.completed_cycles, bool) or not isinstance(self.completed_cycles, int):
            raise ValueError("完成周期数必须是整数")
        if self.completed_cycles < 0:
            raise ValueError("完成周期数必须是非负整数")
