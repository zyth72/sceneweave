"""System3：把意向落实为本次完整主动行为集合。

System3 输出从现在起要主动实施或维持的全部行为：维持既有行动、明确
修改未来部分、或提交一组新的并行行动；显式空集合表示撤回全部主动
维持。发现只是当前做法不可行可以要求重做（须说明原因）；发现意向
本身无法落实则反馈 System2。语义可执行性由 World 检查，这里不做
固定动作表判断。
"""

from sceneweave.domain.identity import ResponseId
from sceneweave.domain.tracing import TurnTracer
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import OwnActionDigest, TurnFacts, trace_stage
from sceneweave.roles.model_output import ModelOutputError, System3Plan, parse_system3_plan

SCHEMA = (
    '{"feasible": true/false, "reason": "不可行或重做的原因", "retry": true/false,'
    ' "maintain": ["继续维持的行动身份", ...],'
    ' "modify": [{"id": "行动身份", "change": "明确的修改"}, ...],'
    ' "new_actions": ["新行动的自然语言描述", ...]}'
)


def build_prompt(
    facts: TurnFacts,
    intention: str,
    own_actions: tuple[OwnActionDigest, ...],
    context_note: str | None,
    abandon: tuple[str, ...],
    attempt: int,
    background: str,
    memory: tuple[str, ...],
) -> str:
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
        f"当前意向：{intention}",
        "私人认知：",
        *([f"- {item}" for item in memory] or ["- 无"]),
        "本轮感知（原始已知观测）：",
        *observations,
        "自身行动状态（引擎确认的细节）：",
        *known,
        "自身可继续主动维持的行动：",
        *actions,
    ]
    if abandon:
        lines.append(f"System2 决定放弃以下行动的意向：{list(abandon)}")
    if context_note is not None:
        lines.append(f"需要针对的问题：{context_note}")
    if attempt > 1:
        lines.append(f"这是第 {attempt} 次落实尝试。")
    lines += [
        "第三步：给出从现在起你要主动实施或维持的完整行为集合。",
        "维持既有行动就列出身份；要调整未来部分就写明修改；新尝试构成一个并行组；",
        "maintain 与 modify 互斥：同一个行动身份只能出现在其中一处。"
        "原样继续已有尝试只填 maintain，不因尚未成功或再次说明目标就重复填 modify；",
        "不再主动维持任何行为时用全空的集合显式表达。",
        "仅撤回某个主动维持时，把该行动身份从 maintain 与 modify 中省略即可，完整集合的省略交由 World 结算；"
        "modify 只用于调整仍在主动维持行动的未来做法，不要用修改描述“放弃/停止维持”来模拟撤回；",
        "你可以在撤回旧动作的同一份完整声明中提出新尝试，也可以保留其它仍想维持的行为，不被迫一并放弃。",
        "本次新提交只能是当前就能独立结算的一步（一个行动或一个并行组）；依赖某一步实际结果的后续步骤，"
        "必须等那一步的反馈确认后，在之后的轮次再决定。",
        "不能把“走过去并拿走物品”这类串行链塞进一条新行动，也不能拆成同组多个行动来掩盖依赖；"
        "不能在描述里预设自己已经成功，也不能替其他主体的自主反应作决定。",
        "例如你距离想拿取的物品还有约 1.4 米、尚不可触及时，先提交“靠近到可触及位置”这一步，"
        "等待到达反馈后再在另一轮决定是否拿取；已经站在可触及时，直接提交拿取尝试即可。",
        "三种输出互斥：正式声明为 feasible=true、retry=false，携带完整行动声明（maintain/modify/new_actions）；"
        "内部重做为 feasible=true、retry=true；意向不可行为 feasible=false、retry=false。"
        "后两种 reason 必填，且 maintain/modify/new_actions 必须全为空；feasible=false 不能同时 retry=true。",
        "retry 或 feasible=false 是内部反馈，不是向引擎提交空集合，也不会取消已接受行动；"
        "若你已明确修正做法，请直接输出正式 feasible=true、retry=false 的完整声明，不要用 retry=true 再携带行动。",
        "只输出一个 JSON 对象，只含 schema 字段，四个必填键一个都不能省：",
        SCHEMA,
    ]
    return "\n".join(lines)


async def run(
    sampler: TextSampler,
    facts: TurnFacts,
    intention: str,
    own_actions: tuple[OwnActionDigest, ...],
    *,
    response_id: ResponseId,
    context_note: str | None = None,
    abandon: tuple[str, ...] = (),
    attempt: int = 1,
    background: str = "",
    memory: tuple[str, ...] = (),
    tracer: TurnTracer | None = None,
) -> System3Plan:
    """执行 System3；解析失败与不可行反馈如实传播。"""
    prompt = build_prompt(
        facts, intention, own_actions, context_note, abandon, attempt, background, memory
    )
    try:
        raw = await sampler.sample(prompt, component="system3")
    except Exception as error:
        trace_stage(
            tracer,
            "system3.declare",
            facts,
            response_id,
            "System3 采样失败",
            prompt=prompt,
            raw_output=None,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    try:
        plan = parse_system3_plan(raw)
    except ModelOutputError as error:
        trace_stage(
            tracer,
            "system3.declare",
            facts,
            response_id,
            "System3 输出未通过结构校验",
            prompt=prompt,
            raw_output=raw,
            error=str(error),
        )
        raise
    trace_stage(
        tracer,
        "system3.declare",
        facts,
        response_id,
        "System3 给出完整声明"
        if (plan.feasible and not plan.retry)
        else ("System3 要求重做" if plan.retry else "System3 反馈意向不可行"),
        prompt=prompt,
        raw_output=raw,
        details={
            "feasible": plan.feasible,
            "retry": plan.retry,
            "reason": plan.reason,
            "maintain": "、".join(str(item) for item in plan.maintain),
            "modify": "；".join(f"{item.action_id}:{item.change}" for item in plan.modify),
            "new_actions": "；".join(plan.new_descriptions),
            "attempt": attempt,
        },
    )
    return plan
