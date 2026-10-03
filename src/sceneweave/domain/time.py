"""剧情时间表示与入口转换。

剧情时间统一为非负整数毫秒；配置中的秒只能在入口转换一次，
下游模块不得使用浮点时间或自行解释 tick。毫秒是存储分辨率，
不表示每毫秒调用模型。
"""

from decimal import Decimal
from typing import NewType

SimulationTime = NewType("SimulationTime", int)


def validate_simulation_time(value: int, field: str) -> None:
    """毫秒时间必须是非负整数；bool 不是时间。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}必须是非负整数毫秒，得到：{value!r}")
    if value < 0:
        raise ValueError(f"{field}必须是非负整数毫秒，得到：{value!r}")


def milliseconds_from_seconds(seconds: int | float) -> SimulationTime:
    """把配置中的剧情秒转换为整数毫秒；这是秒进入系统的唯一边界。

    只接受 int 或 float（bool 与字符串等其他类型直接拒绝）；转换经
    十进制元组精确移位，不依赖 Decimal 上下文精度，大值不会默默舍入。
    无法用整毫秒表示、负数与非有限值拒绝；场景与调度内部只使用毫秒。
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError(f"秒配置必须是 int 或 float，得到：{seconds!r}")
    exact = Decimal(str(seconds))
    if not exact.is_finite():
        raise ValueError(f"秒配置必须是非负有限数，得到：{seconds!r}")
    sign, digits, exponent = exact.as_tuple()
    if not isinstance(exponent, int):  # pragma: no cover - is_finite 已排除
        raise ValueError(f"秒配置必须是非负有限数，得到：{seconds!r}")
    if exponent + 3 < 0:
        raise ValueError(f"秒配置无法用整毫秒表示：{seconds!r}")
    # 直接在十进制元组上移位构造，避免上下文精度引起的静默舍入。
    shifted = Decimal((sign, digits, exponent + 3))
    if shifted < 0:
        raise ValueError(f"秒配置必须是非负数，得到：{seconds!r}")
    return SimulationTime(int(shifted))
