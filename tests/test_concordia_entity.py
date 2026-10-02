"""以真实 Concordia 接口和假实体检查接入，不访问模型 API。"""

import pytest
from concordia.typing.entity import ActionSpec, Entity, OutputType

from sceneweave.domain.messages import EventId, Observation, SimulationTime
from sceneweave.integrations.concordia.entity import ConcordiaEntityAdapter


class ScriptedEntity(Entity):
    def __init__(self, candidate: str) -> None:
        self.candidate = candidate
        self.observations: list[str] = []

    @property
    def name(self) -> str:
        return "测试角色"

    def observe(self, observation: str) -> None:
        self.observations.append(observation)

    def act(self, action_spec: ActionSpec | None = None) -> str:
        return self.candidate


def test_adapter_delivers_observations_and_returns_only_candidate_text() -> None:
    entity = ScriptedEntity("等待")
    adapter = ConcordiaEntityAdapter(entity)
    adapter.observe((Observation(EventId("event-1"), SimulationTime(0), "门仍然关闭"),))
    candidate = adapter.sample_action(
        ActionSpec(call_to_action="本轮选择", output_type=OutputType.CHOICE, options=("等待",))
    )
    assert entity.observations == ["门仍然关闭"]
    assert candidate == "等待"


def test_adapter_rejects_output_that_violates_action_spec() -> None:
    adapter = ConcordiaEntityAdapter(ScriptedEntity("不存在的选项"))
    with pytest.raises(ValueError):
        adapter.sample_action(
            ActionSpec(call_to_action="本轮选择", output_type=OutputType.CHOICE, options=("等待",))
        )
