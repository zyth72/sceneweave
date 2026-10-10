"""System2：决定意向。

System2 只产出当前主观意向：想达成什么，或者保持、改变原有意向。
它不负责物理可执行性，也不输出行动身份或放弃指令——具体做什么、
能否做到由 System3 与 World 负责。World 打回原因与 System3 的不可行
反馈都作为上下文传入，供重新选择参考。
"""

from sceneweave.domain.identity import ResponseId
from sceneweave.domain.tracing import TurnTracer
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import OwnActionDigest, TurnFacts, trace_stage
from sceneweave.roles.model_output import (
    ModelOutputError,
    System2Intention,
    parse_system2_intention,
)

SCHEMA = '{"intention": "非空的当前意向"}'


def build_prompt(
    facts: TurnFacts,
    memory: tuple[str, ...],
    own_actions: tuple[OwnActionDigest, ...],
    previous_intention: str | None,
    feedback_reason: str | None,
    rejection_reason: str | None,
    background: str,
) -> str:
    notes = [f"- {item}" for item in memory] or ["- 无"]
    actions = [f"- {aid}（{phase}）：{description}" for aid, phase, description in own_actions] or [
        "- 无"
    ]
    observations = [f"- {item}" for item in facts.observations] or ["- 无外部感知"]
    known = [f"- {aid}（{phase}）：{detail}" for aid, phase, detail in facts.known_actions] or [
        "- 无"
    ]
    lines = [
        f"你是角色 {facts.role_id}。{background}".rstrip(),
        f"当前剧情时刻 {facts.simulation_time} 毫秒。",
        "本轮感知（原始已知观测）：",
        *observations,
        "自身行动状态（引擎确认的细节）：",
        *known,
        "私人认知（近期记忆）：",
        *notes,
        "自身仍在主动维持的行动：",
        *actions,
    ]
    if previous_intention is not None:
        lines.append(f"上一个意向：{previous_intention}")
    if rejection_reason is not None:
        lines.append(f"上一份完整声明被检查打回：{rejection_reason}")
    if feedback_reason is not None:
        lines.append(f"落实阶段反馈该意向无法落实：{feedback_reason}")
    if rejection_reason is not None or feedback_reason is not None:
        lines.append("请据此重新选择或调整意向。")
    lines += [
        "第二步：决定你现在的意向（保持、改变或放弃）。",
        "你只负责想要什么：不宣布世界里已经发生什么，也不决定能不能做到、具体怎么做；"
        "物理可执行性与行动身份由后续阶段负责。",
        "只输出一个 JSON 对象，只含 intention 一个字段，格式：",
        SCHEMA,
    ]
    return "\n".join(lines)


async def run(
    sampler: TextSampler,
    facts: TurnFacts,
    memory: tuple[str, ...],
    own_actions: tuple[OwnActionDigest, ...],
    *,
    response_id: ResponseId,
    previous_intention: str | None = None,
    feedback_reason: str | None = None,
    rejection_reason: str | None = None,
    background: str = "",
    tracer: TurnTracer | None = None,
) -> System2Intention:
    """执行 System2；意向选择失败如实传播。"""
    prompt = build_prompt(
        facts,
        memory,
        own_actions,
        previous_intention,
        feedback_reason,
        rejection_reason,
        background,
    )
    try:
        raw = await sampler.sample(prompt, component="system2")
    except Exception as error:
        trace_stage(
            tracer,
            "system2.intend",
            facts,
            response_id,
            "System2 采样失败",
            prompt=prompt,
            raw_output=None,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    try:
        intention = parse_system2_intention(raw)
    except ModelOutputError as error:
        trace_stage(
            tracer,
            "system2.intend",
            facts,
            response_id,
            "System2 输出未通过结构校验",
            prompt=prompt,
            raw_output=raw,
            error=str(error),
        )
        raise
    trace_stage(
        tracer,
        "system2.intend",
        facts,
        response_id,
        "System2 选定意向",
        prompt=prompt,
        raw_output=raw,
        details={"intention": intention.intention},
    )
    return intention
