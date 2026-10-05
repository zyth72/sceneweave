"""运行配置：内思考间隔与最长剧情时长。

配置中的秒只能经 from_seconds 在入口转换为整数毫秒；两个值都必须
是正数（正间隔与正时长），bool 不被当作整数接受。
"""

from dataclasses import dataclass

from sceneweave.domain.time import SimulationTime, milliseconds_from_seconds


def _require_positive(value: SimulationTime, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}必须是正整数毫秒，得到：{value!r}")
    if value <= 0:
        raise ValueError(f"{field}必须是正整数毫秒，得到：{value!r}")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """本次运行的调度参数；从运行起始剧情时刻开始计算。"""

    introspection_interval: SimulationTime
    max_duration: SimulationTime

    def __post_init__(self) -> None:
        _require_positive(self.introspection_interval, "内思考间隔")
        _require_positive(self.max_duration, "最长剧情时长")

    @classmethod
    def from_seconds(
        cls,
        introspection_interval_seconds: int | float,
        max_simulation_duration_seconds: int | float,
    ) -> RunConfig:
        """秒配置的唯一入口；无法用整毫秒表示或非正数直接拒绝。"""
        return cls(
            introspection_interval=milliseconds_from_seconds(introspection_interval_seconds),
            max_duration=milliseconds_from_seconds(max_simulation_duration_seconds),
        )

    def limit_from(self, start_time: SimulationTime) -> SimulationTime:
        """运行时限：起始剧情时刻加最长时长；边界时刻的事件仍被处理。"""
        return SimulationTime(start_time + self.max_duration)
