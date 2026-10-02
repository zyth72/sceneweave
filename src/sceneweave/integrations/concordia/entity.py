"""锁定版本的 Concordia Entity 接入，仅处理感知输入和候选输出。"""

from concordia.typing.entity import ActionSpec, Entity

from sceneweave.domain.messages import Observation


class ConcordiaEntityAdapter:
    """将本项目的感知传给实体，保留模型输出与行动提交的边界。

    接口是同步的，可能调用真实模型。角色负责决定是否调用及解析候选输出；
    本适配器不实现 System1/2/3，不直接生成 RoleResponse 或提交世界事实。
    """

    def __init__(self, entity: Entity) -> None:
        self._entity = entity

    def observe(self, observations: tuple[Observation, ...]) -> None:
        """传递已经由 World 过滤、并由角色确认属于自身的感知。"""
        for observation in observations:
            self._entity.observe(observation.content)

    def sample_action(self, action_spec: ActionSpec) -> str:
        """获得并检查候选文本格式；格式成立不代表行动可行或已经成功。"""
        candidate = self._entity.act(action_spec=action_spec)
        action_spec.validate(candidate)
        return candidate
