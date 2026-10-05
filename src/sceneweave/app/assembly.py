"""完整装配：从配置构建 ModelRole + ModelTranslator + ModelWorld + Scheduler。

作用域命名空间：为避免角色 ID 恰好是 "world-progress" 等造成的碰撞，
全部使用带前缀的内部作用域——角色 `role:{id}`、翻译器 `translator:{id}`、
World 检查 `world:check:{id}`、World 进展 `world:progress`、World 裁决
`world:adjudicate`。World 内部使用 `world-progress` 等裸名，装配层
负责映射到带命名空间的采样器工厂调用。
"""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from concordia.language_model import language_model

from sceneweave.config.loader import AssemblyConfig
from sceneweave.domain.identity import EventId, RoleId
from sceneweave.domain.messages import WorldEvent
from sceneweave.domain.modeling import TextSampler
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TurnTracer
from sceneweave.engine.control import RunGate, RunStatus
from sceneweave.engine.scheduler import Scheduler
from sceneweave.integrations.concordia.sampling import LanguageModelSampler
from sceneweave.roles.model_role import ModelRole
from sceneweave.translator.model_translator import ModelTranslator
from sceneweave.world.model_world import ModelWorld


class SamplerFactory(Protocol):
    """采样器工厂：按内部作用域返回采样器，每作用域只调用一次。"""

    def __call__(self, scope: str) -> TextSampler: ...


class AssemblyError(ValueError):
    """装配配置或工厂错误。"""


@dataclass(frozen=True, slots=True)
class Assembly:
    """装配结果：公开引用供 runner/测试使用。"""

    scheduler: Scheduler
    world: ModelWorld
    roles: tuple[ModelRole, ...]
    translator: ModelTranslator


def _import_live_factory(
    spec: str,
) -> Callable[[str, dict[str, Any]], language_model.LanguageModel]:
    if ":" not in spec:
        raise AssemblyError(
            f"live 工厂必须是 module:callable 格式，得到：{spec!r}。"
            "工厂签名 (scope, options) -> LanguageModel"
        )
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise AssemblyError(f"live 工厂格式无效：{spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise AssemblyError(f"无法导入 live 工厂模块 {module_name!r}：{error}") from error
    factory = getattr(module, attr, None)
    if factory is None or not callable(factory):
        raise AssemblyError(f"live 工厂不可调用：{spec!r}")
    from typing import cast

    return cast("Callable[[str, dict[str, Any]], language_model.LanguageModel]", factory)


def build_live_factory(
    module_callable: str,
    options: Mapping[str, Any],
) -> Callable[[str], TextSampler]:
    """构建 live 模式采样器工厂（严格类型校验，不静默 fallback）。

    返回的采样器是 LanguageModelSampler；组件日志通道由运行层
    （runner）经 bind_channel_for 注入，使全部条目进入同一
    SimulationLog——这里不自行创建日志。
    """
    raw_factory = _import_live_factory(module_callable)
    opts = dict(options)
    cache: dict[str, LanguageModelSampler] = {}

    def factory(scope: str) -> TextSampler:
        if scope in cache:
            return cache[scope]
        model = raw_factory(scope, opts)
        if not isinstance(model, language_model.LanguageModel):
            raise AssemblyError(
                f"live 工厂返回类型必须是 LanguageModel（scope={scope}），"
                f"得到：{type(model).__name__}"
            )
        sampler = LanguageModelSampler(model)
        cache[scope] = sampler
        return sampler

    return factory


def build_scripted_factory(
    scripts: Mapping[str, Mapping[str, list[str]]],
) -> Callable[[str], TextSampler]:
    """构建脚本模式采样器工厂（每作用域独立 ScriptedModel）。"""
    from sceneweave.app.scripted import ScriptedModel

    cache: dict[str, ScriptedModel] = {}

    def factory(scope: str) -> TextSampler:
        if scope in cache:
            return cache[scope]
        cache[scope] = ScriptedModel(dict(scripts.get(scope, {})))
        return cache[scope]

    return factory


def world_to_namespaced(world_scope: str) -> str:
    """把 World 内部作用域名映射到带命名空间的采样器作用域名。

    world-progress → world:progress
    world-adjudicate → world:adjudicate
    world-check-{role} → world:check:{role}
    """
    if world_scope == "world-progress":
        return "world:progress"
    if world_scope == "world-adjudicate":
        return "world:adjudicate"
    if world_scope.startswith("world-check-"):
        return f"world:check:{world_scope.removeprefix('world-check-')}"
    raise AssemblyError(f"未知的 World 采样作用域：{world_scope}")


def _world_scope_mapper(factory: SamplerFactory) -> Callable[[str], TextSampler]:
    def mapped(world_scope: str) -> TextSampler:
        return factory(world_to_namespaced(world_scope))

    return mapped


def assemble(
    config: AssemblyConfig,
    sampler_factory: SamplerFactory,
    *,
    tracer: TurnTracer | None = None,
    should_end: Callable[[RunStatus], bool] | None = None,
    gate: RunGate | None = None,
) -> Assembly:
    """从配置与采样器工厂构建完整装配。

    角色私人 background 只进该角色的 ModelRole，不进入 World 或其他
    任何作用域的提示。should_end/gate 经 Scheduler 公开构造传入。
    """
    world = ModelWorld(
        scene=config.scene,
        sampler_for_scope=_world_scope_mapper(sampler_factory),
        roles=tuple(RoleId(spec.role_id) for spec in config.roles),
        tracer=tracer,
    )

    roles = tuple(
        ModelRole(
            RoleId(spec.role_id),
            sampler_factory(f"role:{spec.role_id}"),
            tracer=tracer,
            background=spec.background,
        )
        for spec in config.roles
    )

    def translator_factory(role_id: RoleId) -> TextSampler:
        return sampler_factory(f"translator:{role_id}")

    translator = ModelTranslator(translator_factory, tracer=tracer)

    initial_events = tuple(
        WorldEvent(EventId(event.event_id), SimulationTime(event.time_ms), event.description)
        for event in config.initial_events
    )

    scheduler = Scheduler(
        roles=roles,
        world=world,
        translator=translator,
        config=config.run.to_run_config(),
        initial_events=initial_events,
        tracer=tracer,
        gate=gate,
        should_end=should_end,
    )
    return Assembly(scheduler=scheduler, world=world, roles=roles, translator=translator)
