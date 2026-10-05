"""System1：判断这件事是否影响我、是否需要重新考虑。

System1 只更新自身认知并给出是否重新考虑的判断：认识更新不等于必须
行动，不重新考虑时也不清空既有维持集合（由角色组装完整维持声明）。
"""

from sceneweave.domain.identity import ResponseId
from sceneweave.domain.tracing import TurnTracer
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import TurnFacts, trace_stage
from sceneweave.roles.model_output import ModelOutputError, System1Decision, parse_system1_decision

SCHEMA = '{"reconsider": true/false, "note": "非空的认知更新说明"}'


def build_prompt(facts: TurnFacts, memory: tuple[str, ...], background: str) -> str:
    observations = [f"- {item}" for item in facts.observations] or ["- 无外部感知"]
    actions = [f"- {aid}（{phase}）：{detail}" for aid, phase, detail in facts.known_actions] or [
        "- 无"
    ]
    notes = [f"- {item}" for item in memory] or ["- 无"]
    lines = [
        f"你是角色 {facts.role_id}。{background}".rstrip(),
        f"当前剧情时刻 {facts.simulation_time} 毫秒。",
        f"触发事件：{[str(e) for e in facts.event_ids] or '无'}"
        + ("（含周期性内思考）" if facts.introspection else ""),
        "本轮感知（原始已知观测）：",
        *observations,
        "自身行动状态（引擎确认的细节）：",
        *actions,
        "私人认知（近期记忆）：",
        *notes,
        "第一步：判断这些信息是否影响你、是否需要重新考虑当前做法。",
        "只输出一个 JSON 对象，只含这两个字段，格式：",
        SCHEMA,
    ]
    return "\n".join(lines)


async def run(
    sampler: TextSampler,
    facts: TurnFacts,
    memory: tuple[str, ...],
    *,
    response_id: ResponseId,
    background: str = "",
    tracer: TurnTracer | None = None,
) -> System1Decision:
    """执行 System1；解析失败如实传播，不转换为空或默认判断。"""
    prompt = build_prompt(facts, memory, background)
    try:
        raw = await sampler.sample(prompt, component="system1")
    except Exception as error:
        trace_stage(
            tracer,
            "system1.decide",
            facts,
            response_id,
            "System1 采样失败",
            prompt=prompt,
            raw_output=None,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    try:
        decision = parse_system1_decision(raw)
    except ModelOutputError as error:
        trace_stage(
            tracer,
            "system1.decide",
            facts,
            response_id,
            "System1 输出未通过结构校验",
            prompt=prompt,
            raw_output=raw,
            error=str(error),
        )
        raise
    trace_stage(
        tracer,
        "system1.decide",
        facts,
        response_id,
        "System1 判断完成",
        prompt=prompt,
        raw_output=raw,
        details={"reconsider": decision.reconsider, "note": decision.note},
    )
    return decision
