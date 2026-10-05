"""引擎事件队列的消息类型。

队列保存完整事件内容：世界事件带描述，行动节点事件带行动身份与
类型，内思考事件单独标记。调度只按最近时刻跳转并在同刻整批取出，
批内顺序按事件身份稳定排序，插入顺序不构成剧情优先级。
"""

import enum
import heapq
from dataclasses import dataclass

from sceneweave.domain.identity import (
    ActionId,
    EventId,
    require_identifier,
    require_text,
    require_unique_identifiers,
)
from sceneweave.domain.time import SimulationTime, validate_simulation_time


class ScheduledEventKind(enum.StrEnum):
    """引擎事件的类型；行动开始、过程与结算是不同的调度边界。"""

    WORLD = "world"
    ACTION_START = "action_start"
    ACTION_CHECKPOINT = "action_checkpoint"
    ACTION_END = "action_end"
    INTROSPECTION = "introspection"


_ACTION_KINDS = frozenset(
    {
        ScheduledEventKind.ACTION_START,
        ScheduledEventKind.ACTION_CHECKPOINT,
        ScheduledEventKind.ACTION_END,
    }
)


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """队列中的一个待处理事件；身份在待处理集合内唯一，内容完整保留。"""

    event_id: EventId
    simulation_time: SimulationTime
    kind: ScheduledEventKind
    description: str
    action_id: ActionId | None = None

    def __post_init__(self) -> None:
        require_identifier(self.event_id, "事件身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.kind, ScheduledEventKind):
            raise ValueError("事件类型必须是 ScheduledEventKind")
        require_text(self.description, "事件描述")
        if self.kind in _ACTION_KINDS:
            if self.action_id is None:
                raise ValueError("行动节点事件必须携带行动身份")
        elif self.action_id is not None:
            raise ValueError("非行动事件不得携带行动身份")


class EventQueue:
    """按最近时刻跳转的事件队列；同刻事件整批取出。

    内部用堆维护时间顺序；事件身份在待处理集合内唯一，重复登记直接
    拒绝。已被取出或移除的身份属于历史，不阻止同名事件的再次登记
    （例如修改后的行动节点按新安排重新入队）。批内输出按事件身份
    排序，与插入顺序无关。
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, ScheduledEvent]] = []
        self._sequence = 0
        self._live: dict[EventId, ScheduledEvent] = {}

    def __len__(self) -> int:
        return len(self._live)

    def schedule(self, event: ScheduledEvent) -> None:
        if not isinstance(event, ScheduledEvent):
            raise ValueError("队列只能登记 ScheduledEvent")
        if event.event_id in self._live:
            raise ValueError(f"事件身份已存在：{event.event_id}")
        self._live[event.event_id] = event
        heapq.heappush(self._heap, (event.simulation_time, self._sequence, event))
        self._sequence += 1

    def next_time(self) -> SimulationTime | None:
        self._prune()
        if not self._heap:
            return None
        return SimulationTime(self._heap[0][0])

    def take_batch(self, now: SimulationTime) -> tuple[ScheduledEvent, ...]:
        """取出时刻恰好等于 now 的全部事件，按事件身份排序。"""
        validate_simulation_time(now, "剧情时刻")
        if self.next_time() != now:
            raise ValueError(f"时刻 {now} 没有待处理事件批")
        batch: list[ScheduledEvent] = []
        while self._heap and self._heap[0][0] == now:
            _, _, event = heapq.heappop(self._heap)
            if event.event_id in self._live:
                batch.append(event)
                del self._live[event.event_id]
        return tuple(sorted(batch, key=lambda item: str(item.event_id)))

    def remove_for_action(self, action_id: ActionId) -> tuple[ScheduledEvent, ...]:
        """移除某行动的全部待处理事件；已发生的历史节点早已出队。"""
        require_identifier(action_id, "行动身份")
        removed = [event for event in self._live.values() if event.action_id == action_id]
        if removed:
            for event in removed:
                del self._live[event.event_id]
            removed_ids = {event.event_id for event in removed}
            self._heap = [entry for entry in self._heap if entry[2].event_id not in removed_ids]
            heapq.heapify(self._heap)
        return tuple(sorted(removed, key=lambda item: (item.simulation_time, str(item.event_id))))

    def snapshot(self) -> tuple[ScheduledEvent, ...]:
        """当前待处理事件的稳定快照，按（时刻，事件身份）排序。"""
        return tuple(
            sorted(
                self._live.values(),
                key=lambda item: (item.simulation_time, str(item.event_id)),
            )
        )

    def _prune(self) -> None:
        while self._heap and self._heap[0][2].event_id not in self._live:
            heapq.heappop(self._heap)


def validate_unique_event_ids(events: list[ScheduledEvent]) -> None:
    """批量校验待处理事件身份唯一；供构造入口一次报出全部问题。"""
    require_unique_identifiers([event.event_id for event in events], "事件身份")
