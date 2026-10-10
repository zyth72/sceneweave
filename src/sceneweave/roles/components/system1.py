"""System1：判断这件事是否影响我、是否需要重新考虑。

System1 只更新自身认知并给出三路分流：``continue``（沿用已保存计划）、
``replan``（保留意向、只让 System3 重新落实）、``rethink``（重选意向）。
认识更新不等于必须行动，``continue`` 也不清空既有维持集合。System1 只看
本角色自身状态：当前意向与已保存计划摘要，不拿其他角色私有信息。
"""

from sceneweave.domain.identity import ResponseId
from sceneweave.domain.tracing import TurnTracer
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import OwnActionDigest, TurnFacts, trace_stage
from sceneweave.roles.model_output import ModelOutputError, System1Decision, parse_system1_decision

SCHEMA = '{"route": "continue|replan|rethink", "note": "非空的认知更新说明"}'


def build_prompt(
    facts: TurnFacts,
    memory: tuple[str, ...],
    background: str,
    intention: str | None,
    plan_digest: tuple[OwnActionDigest, ...],
) -> str:
    observations = [f"- {item}" for item in facts.observations] or ["- 无外部感知"]
    actions = [f"- {aid}（{phase}）：{detail}" for aid, phase, detail in facts.known_actions] or [
        "- 无"
    ]
    notes = [f"- {item}" for item in memory] or ["- 无"]
    plan_lines = [f"- {aid}（{phase}）：{description}" for aid, phase, description in plan_digest]
    lines = [
        f"你是角色 {facts.role_id}。{background}".rstrip(),
        f"当前剧情时刻 {facts.simulation_time} 毫秒。",
        f"触发事件：{[str(e) for e in facts.event_ids] or '无'}"
        + ("（含周期性内思考）" if facts.introspection else ""),
        "本轮感知（原始已知观测）：",
        *observations,
        "自身行动状态（引擎确认的细节）：",
        *actions,
        f"你当前已保存的意向：{intention if intention is not None else '（无，尚未决定过意向）'}",
        "你已保存的完整动作链与时间安排（本角色私有；continue 将沿用它们）：",
        *(plan_lines or ["- 无（尚无已保存计划）"]),
        "私人认知（近期记忆）：",
        *notes,
        "第一步：判断这些信息是否影响你，并选择三路分流之一：",
        "- continue：沿用你已保存的完整动作链与时间安排，只推进最近步骤；"
        "不重选意向、不重新落实。没有已保存计划时不要选它。",
        "- replan：保留当前意向，只让 System3 按新的已知条件重新落实完整动作链。"
        "没有当前意向时不要选它。",
        "- rethink：重新选择意向（System2）后再由 System3 落实。"
        "第一次决定意向、或原来的目标已经不再合适时必须选它。",
        "没有可行动的信息、没有明确目标或愿意先等待时，若已有保存计划可选 continue；"
        "否则选 rethink 明确一个意向。",
        "你只能看到自己的意向、计划与状态，看不到其他角色的私有认知。",
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
    intention: str | None = None,
    plan_digest: tuple[OwnActionDigest, ...] = (),
    background: str = "",
    tracer: TurnTracer | None = None,
) -> System1Decision:
    """执行 System1；解析失败如实传播，不转换为空或默认判断。"""
    prompt = build_prompt(facts, memory, background, intention, plan_digest)
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
        details={"route": decision.route.value, "note": decision.note},
    )
    return decision
