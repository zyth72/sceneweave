"""System1 程序路由策略：从合法倾向分布与本角色结构状态决定最终 route。

模型只给出三路倾向；最终 route 由这里的程序策略决定，选择方式、允许
路径、归一化权重与实际随机抽值全部留给调用方留痕。策略只看本角色自身
结构状态（是否已有意向、是否有整链检查通过并保存的计划），不读 World
未感知的客观状态，也不替 System3 判断是否有方案。

结构规则：
- 没有意向：强制 rethink（结构强制），记录 forced 原因；只会先校验分布。
- 有意向但没有已保存计划：continue 不合法，屏蔽后重新归一化；剩余权重
  为 0 明确报错，不猜测 fallback。
- 有意向且有已保存计划：三路均合法（已保存的 [] 是有效空计划）。

确定性规则：合法归一化分布存在唯一最大项且（最大权重 >= 确定阈值 或
最大与次大之差 >= 边际阈值）时确定选择；否则按合法归一化分布采样。并列
最大一律进入采样，不用遍历顺序打平局。

随机源在角色边界显式提供：每个角色由 ``derive_route_rng`` 依据 route_seed
与角色身份稳定散列派生独立 ``random.Random``，避免并发角色调用次序改变
随机序列；只有在采样分支才消耗随机数，相同种子与策略可复现相同路由。
"""

import hashlib
import math
import random
from collections.abc import Mapping

from sceneweave.roles.model_output import (
    SYSTEM1_METHOD_DETERMINISTIC,
    SYSTEM1_METHOD_FORCED,
    SYSTEM1_METHOD_SAMPLED,
    ModelOutputError,
    System1Decision,
    System1Route,
    System1RoutePolicy,
    System1Tendency,
)

# 默认路由种子：显式固定值，便于复现；可由 ModelRole 参数覆盖。
DEFAULT_SYSTEM1_ROUTE_SEED = 0


def allowed_routes(*, has_intention: bool, has_saved_plan: bool) -> tuple[System1Route, ...]:
    """按本角色结构状态给出合法路径（不含强制规则，供策略与留痕共用）。"""
    if not has_intention:
        # 没有意向时结构强制 rethink；这里返回强制结果，便于统一留痕。
        return (System1Route.RETHINK,)
    if not has_saved_plan:
        return (System1Route.REPLAN, System1Route.RETHINK)
    return (System1Route.CONTINUE, System1Route.REPLAN, System1Route.RETHINK)


def derive_route_rng(route_seed: int, role_id: str) -> tuple[random.Random, str]:
    """按 (route_seed, 角色身份) 稳定散列派生每角色独立的随机源。

    返回 (随机源, 种子散列十六进制)。散列写入留痕，足以复核复现，不使用
    全局 random，也不使用墙钟种子。
    """
    if type(route_seed) is bool or not isinstance(route_seed, int):
        raise ValueError(f"route_seed 必须是整数，得到：{route_seed!r}")
    if not isinstance(role_id, str) or not role_id:
        raise ValueError("角色身份必须是非空字符串")
    material = f"sceneweave-system1-route|{route_seed}|{role_id}".encode()
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest, "big")), digest.hex()


def decide_route(
    tendency: System1Tendency,
    *,
    has_intention: bool,
    has_saved_plan: bool,
    rng: random.Random,
    policy: System1RoutePolicy,
    route_seed: int,
    seed_hash: str,
) -> System1Decision:
    """由倾向分布与本角色结构状态决定最终 route；不做任何模型调用。"""
    normalized = tendency.normalized()
    if not has_intention:
        # 结构强制：忽略模型分布的零质量，但仍然是在分布通过校验之后。
        return _build(
            route=System1Route.RETHINK,
            tendency=tendency,
            normalized=normalized,
            allowed=(System1Route.RETHINK,),
            legal={System1Route.RETHINK: 1.0},
            method=SYSTEM1_METHOD_FORCED,
            reason="no_intention",
            max_weight=1.0,
            margin=0.0,
            random_value=None,
            policy=policy,
            route_seed=route_seed,
            seed_hash=seed_hash,
        )
    allowed = allowed_routes(has_intention=True, has_saved_plan=has_saved_plan)
    allowed_sum = math.fsum(normalized[route] for route in allowed)
    if allowed_sum <= 0.0:
        raise ModelOutputError(
            "允许路径的归一化权重之和为 0（允许 "
            f"{[route.value for route in allowed]}），无法选择且不猜测 fallback"
        )
    legal = {route: normalized[route] / allowed_sum for route in allowed}
    ordered = sorted(legal.values(), reverse=True)
    max_weight = ordered[0]
    margin = ordered[0] - ordered[1]
    winners = [route for route in allowed if legal[route] == max_weight]
    if len(winners) == 1 and (
        max_weight >= policy.certainty_threshold or margin >= policy.margin_threshold
    ):
        return _build(
            route=winners[0],
            tendency=tendency,
            normalized=normalized,
            allowed=allowed,
            legal=legal,
            method=SYSTEM1_METHOD_DETERMINISTIC,
            reason="threshold",
            max_weight=max_weight,
            margin=margin,
            random_value=None,
            policy=policy,
            route_seed=route_seed,
            seed_hash=seed_hash,
        )
    draw = rng.random()
    route = _sample(legal, allowed, draw)
    return _build(
        route=route,
        tendency=tendency,
        normalized=normalized,
        allowed=allowed,
        legal=legal,
        method=SYSTEM1_METHOD_SAMPLED,
        reason="tie" if len(winners) > 1 else "below_threshold",
        max_weight=max_weight,
        margin=margin,
        random_value=draw,
        policy=policy,
        route_seed=route_seed,
        seed_hash=seed_hash,
    )


def _sample(
    legal: Mapping[System1Route, float],
    allowed: tuple[System1Route, ...],
    draw: float,
) -> System1Route:
    """按允许路径的固定顺序抽样；舍入误差只能落到有正权重的路径。"""
    cumulative = 0.0
    for route in allowed:
        cumulative += legal[route]
        if draw < cumulative:
            return route
    return next(route for route in reversed(allowed) if legal[route] > 0.0)


def _build(
    *,
    route: System1Route,
    tendency: System1Tendency,
    normalized: Mapping[System1Route, float],
    allowed: tuple[System1Route, ...],
    legal: Mapping[System1Route, float],
    method: str,
    reason: str,
    max_weight: float,
    margin: float,
    random_value: float | None,
    policy: System1RoutePolicy,
    route_seed: int,
    seed_hash: str,
) -> System1Decision:
    return System1Decision(
        route=route,
        note=tendency.note,
        method=method,
        reason=reason,
        allowed_routes=allowed,
        raw_probabilities=tendency.probabilities,
        normalized_probabilities=normalized,
        legal_weights=legal,
        max_weight=max_weight,
        margin=margin,
        random_value=random_value,
        certainty_threshold=policy.certainty_threshold,
        margin_threshold=policy.margin_threshold,
        route_seed=route_seed,
        seed_hash=seed_hash,
    )
