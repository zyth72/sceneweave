"""System1：判断这件事对本角色状态的影响，给出三路路径倾向。

模型只输出三路**倾向分布**（``continue``/``replan``/``rethink`` 的相对
倾向，不是世界事件概率，也未校准），不直接输出最终 route，也不做随机
选择。最终 route 由 ``system1_policy`` 的程序策略依据本角色结构状态决定：
没有意向强制 rethink；有意向但无已保存计划时 continue 不合法；有意向且
有已保存计划时三路均合法（已保存的 ``[]`` 是有效空计划）。System1 只看
本角色自身状态（当前意向、已保存计划摘要与排程），不拿其他角色私有信息，
也不读取 World 未感知的客观状态。
"""

import random

from sceneweave.domain.identity import ResponseId
from sceneweave.domain.tracing import TraceValue, TurnTracer
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import OwnActionDigest, TurnFacts, trace_stage
from sceneweave.roles.components.system1_policy import (
    DEFAULT_SYSTEM1_ROUTE_SEED,
    allowed_routes,
    decide_route,
    derive_route_rng,
)
from sceneweave.roles.model_output import (
    ModelOutputError,
    System1Decision,
    System1Route,
    System1RoutePolicy,
    parse_system1_tendency,
)

SCHEMA = (
    '{"probabilities": {"continue": 0.34, "replan": 0.33, "rethink": 0.33}, '
    '"note": "非空的认知更新说明"}'
)
_ALL_ROUTE_NAMES = tuple(route.value for route in System1Route)


def _probability_details(decision: System1Decision) -> dict[str, TraceValue]:
    """把程序策略结果转成可留痕的 JSON 安全细节（分布保持嵌套结构）。"""
    return {
        "allowed_routes": [route.value for route in decision.allowed_routes],
        "raw_probabilities": {
            route.value: decision.raw_probabilities[route] for route in System1Route
        },
        "normalized_probabilities": {
            route.value: decision.normalized_probabilities[route] for route in System1Route
        },
        "legal_weights": {
            route.value: decision.legal_weights[route] for route in decision.allowed_routes
        },
        "route": decision.route.value,
        "method": decision.method,
        "reason": decision.reason,
        "max_weight": decision.max_weight,
        "margin": decision.margin,
        "random_value": decision.random_value,
        "certainty_threshold": decision.certainty_threshold,
        "margin_threshold": decision.margin_threshold,
        "route_seed": decision.route_seed,
        "seed_hash": decision.seed_hash,
        "note": decision.note,
    }


def build_prompt(
    facts: TurnFacts,
    memory: tuple[str, ...],
    background: str,
    intention: str | None,
    plan_digest: tuple[OwnActionDigest, ...],
    has_saved_plan: bool,
) -> str:
    observations = [f"- {item}" for item in facts.observations] or ["- 无外部感知"]
    actions = [f"- {aid}（{phase}）：{detail}" for aid, phase, detail in facts.known_actions] or [
        "- 无"
    ]
    notes = [f"- {item}" for item in memory] or ["- 无"]
    plan_lines = [f"- {aid}（{phase}）：{description}" for aid, phase, description in plan_digest]
    if has_saved_plan:
        # 已保存的计划可能是合法空计划：continue 仍然合法，不能说成“缺计划”。
        plan_state = plan_lines or [
            "- 已保存合法空计划：当前没有要主动维持的动作，continue 仍然合法。"
        ]
    else:
        plan_state = ["- 尚无已保存计划：当前不要选择 continue。"]
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
        *plan_state,
        "私人认知（近期记忆）：",
        *notes,
        "第一步：判断这件事对你当前状态的影响层级，给出三路路径倾向。",
        "三路含义：",
        "- continue：沿用已保存的完整动作链与时间安排，只推进最近步骤；"
        "不重选意向、不重新落实。没有已保存计划时不要偏它。",
        "- replan：保留当前意向，只按新的已知条件重新落实完整动作链"
        "（不改目标，只改做法）。没有当前意向时不要偏它。",
        "- rethink：重新选择意向后再落实；第一次决定意向、或原目标已不再合适时必须偏它。",
        "影响层级与倾向的大致对应：事件与当前意向/计划无关或已确认无害 -> 偏 continue；"
        "事件改变了做法可行性或已知条件但仍保留目标 -> 偏 replan；"
        "事件使原目标不再成立、需要重选意向，或你还没有意向 -> 偏 rethink。",
        "你只输出三路倾向，不要给出最终 route，也不要自己做随机选择；最终走哪条路由程序策略决定。",
        "三个数必须在 0 到 1 之间，三者之和约为 1；它们是这次事件影响驱动的相对路径倾向，"
        "不是世界事件发生概率，也不是已校准的置信。",
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
    has_saved_plan: bool = False,
    background: str = "",
    tracer: TurnTracer | None = None,
    policy: System1RoutePolicy | None = None,
    rng: random.Random | None = None,
    route_seed: int = DEFAULT_SYSTEM1_ROUTE_SEED,
    seed_hash: str | None = None,
) -> System1Decision:
    """执行 System1：模型只产倾向，程序策略决定最终 route。

    解析或策略失败如实传播，不转换为默认判断；错误同样留痕，不报成功。
    随机源未显式提供时按 route_seed 与角色身份稳定散列派生，便于复现；
    只有在采样分支才消耗随机数。
    """
    prompt = build_prompt(facts, memory, background, intention, plan_digest, has_saved_plan)
    resolved_policy = policy if policy is not None else System1RoutePolicy()
    resolved_rng = rng
    resolved_seed_hash = seed_hash
    if resolved_rng is None or resolved_seed_hash is None:
        derived_rng, derived_hash = derive_route_rng(route_seed, facts.role_id)
        if resolved_rng is None:
            resolved_rng = derived_rng
        if resolved_seed_hash is None:
            resolved_seed_hash = derived_hash
    has_intention = intention is not None
    context_details: dict[str, TraceValue] = {
        "has_intention": has_intention,
        "has_saved_plan": has_saved_plan,
        "allowed_routes": [
            route.value
            for route in allowed_routes(has_intention=has_intention, has_saved_plan=has_saved_plan)
        ],
        "certainty_threshold": resolved_policy.certainty_threshold,
        "margin_threshold": resolved_policy.margin_threshold,
        "route_seed": route_seed,
        "seed_hash": resolved_seed_hash,
    }
    # 采样请求前先记录本次请求身份：并发角色完成顺序不确定，组件日志
    # 通道在请求开始时捕获的身份必须属于本角色本次响应。
    trace_stage(
        tracer,
        "system1.invoke",
        facts,
        response_id,
        "System1 请求模型给出三路倾向分布",
        prompt=prompt,
        raw_output=None,
        details=context_details,
    )
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
            details=context_details,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    try:
        tendency = parse_system1_tendency(raw)
    except ModelOutputError as error:
        trace_stage(
            tracer,
            "system1.decide",
            facts,
            response_id,
            "System1 输出未通过结构校验",
            prompt=prompt,
            raw_output=raw,
            details=context_details,
            error=str(error),
        )
        raise
    try:
        decision = decide_route(
            tendency,
            has_intention=has_intention,
            has_saved_plan=has_saved_plan,
            rng=resolved_rng,
            policy=resolved_policy,
            route_seed=route_seed,
            seed_hash=resolved_seed_hash,
        )
    except ModelOutputError as error:
        trace_stage(
            tracer,
            "system1.decide",
            facts,
            response_id,
            "System1 程序策略无法选择合法路由",
            prompt=prompt,
            raw_output=raw,
            details={
                **context_details,
                "raw_probabilities": {
                    route.value: tendency.probabilities[route] for route in System1Route
                },
                "normalized_probabilities": {
                    route.value: tendency.normalized()[route] for route in System1Route
                },
                "route_seed": route_seed,
                "seed_hash": resolved_seed_hash,
            },
            error=str(error),
        )
        raise
    details = _probability_details(decision)
    details["has_intention"] = has_intention
    details["has_saved_plan"] = has_saved_plan
    details["probability_sum"] = tendency.total
    trace_stage(
        tracer,
        "system1.decide",
        facts,
        response_id,
        f"System1 完成：程序策略选择 route={decision.route.value}"
        f"（{decision.method}/{decision.reason}）",
        prompt=prompt,
        raw_output=raw,
        details=details,
    )
    return decision
