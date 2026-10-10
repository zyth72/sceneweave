"""模型角色 + 模型翻译器的离线组装演示（脚本假模型，无真实 API）。

演示生产装配形态：ModelRole 的 System1/2/3 与 ModelTranslator 都经
注入采样器获得模型文本；采样器可以是真实 Concordia LanguageModel 桥
（见 integrations/concordia/sampling.py），这里用内联脚本假件保证离线
可运行。结构化日志写入 SimulationLog 并导出 JSON，可离线检索引擎/
世界/角色/翻译的因果链。不依赖未提交的场景卡或 model_trace.py。

运行：

    uv run --locked python examples/model_translator_demo.py
"""

import asyncio
from pathlib import Path

from concordia.utils import structured_logging
from offline_demo import LIN, SHEN, DemoWorld

from sceneweave.domain.identity import EventId
from sceneweave.domain.messages import WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord, TurnTracer
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import Scheduler
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.model_role import ModelRole
from sceneweave.translator.model_translator import ModelTranslator

S1_GO = '{"route": "rethink", "note": "门被推开，需要行动"}'
S1_SKIP = '{"route": "continue", "note": "没有新情况"}'
S2 = '{"intention": "查看书桌上的东西"}'
S3_WALK = '[{"kind": "move", "destination": {"entity": "书桌"}}]'
# 行动身份由角色分配（a-lin_zhou-1）——脚本假件按该身份返回安排。
TRANSLATION = (
    '{"schedules": [{"action_id": "a-lin_zhou-1",'
    ' "start": 2000, "expected_end": 6000, "checkpoints": [4000]}]}'
)


class ScriptedModel:
    """按组件名顺序返回脚本文本的假采样器。"""

    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = {name: list(items) for name, items in script.items()}

    async def sample(self, prompt: str, *, component: str) -> str:
        responses = self.script.get(component)
        if responses is None or not responses:
            raise RuntimeError(f"组件 {component} 的脚本已耗尽")
        return responses.pop(0)


class CompositeTracer(TurnTracer):
    """组合追踪：结构化日志 + 关键边界打印。"""

    INTERESTING = {
        "system3.declare",
        "translator.invoke",
        "translator.parsed",
        "registration.commit",
        "world.commit",
        "engine.stop",
    }

    def __init__(self, log: structured_logging.SimulationLog) -> None:
        self._log_tracer = SimulationLogTracer(log)

    def trace(self, record: TraceRecord) -> None:
        self._log_tracer.trace(record)
        if record.stage in self.INTERESTING:
            head = f"t={record.simulation_time:>6}ms [{record.stage}] {record.summary}"
            if record.outputs:
                head += "：" + "；".join(record.outputs[:3])
            print(head)


async def main() -> None:
    log = structured_logging.SimulationLog()
    tracer = CompositeTracer(log)
    # 生产装配：这里的两个脚本采样器都可替换为
    # LanguageModelSampler(真实模型, channel_for=...) —— 接口一致。
    role_model = ScriptedModel(
        {
            "system1": [S1_GO] + [S1_SKIP] * 6,
            "system2": [S2],
            "system3": [S3_WALK],
        }
    )
    shen_model = ScriptedModel({"system1": [S1_SKIP] * 7})
    translator_model = ScriptedModel({"translator": [TRANSLATION]})
    world = DemoWorld()
    scheduler = Scheduler(
        roles=[
            ModelRole(LIN, role_model, tracer=tracer),
            ModelRole(SHEN, shen_model, tracer=tracer),
        ],
        world=world,
        translator=ModelTranslator.shared(translator_model, tracer=tracer),
        config=RunConfig.from_seconds(
            introspection_interval_seconds=5,
            max_simulation_duration_seconds=10,
        ),
        initial_events=[WorldEvent(EventId("startup"), SimulationTime(0), "书房里很安静")],
        tracer=tracer,
    )
    result = await scheduler.run()
    print()
    print(
        f"运行结束：{result.reason.value}，完成周期 {result.completed_cycles}，"
        f"最终剧情时刻 {result.final_time}ms，林舟位置 {world.position:.2f} 米"
    )
    log_path = Path("model_translator_demo_log.json")
    log_path.write_text(log.to_json(), encoding="utf-8")
    print(f"结构化日志已导出：{log_path}（含 system1/2/3、translator、world、engine 因果链）")
    if result.error:
        print(f"运行错误：{result.error}")


if __name__ == "__main__":
    asyncio.run(main())
