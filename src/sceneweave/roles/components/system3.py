"""System3：把意向落实为一维全量结构化动作计划。

System3 每次输出一个 JSON **一维数组**，完整声明从现在起要做的结构化
原子动作（不是增删改指令）。可选 ``depends_on`` 表示「World 已实际把
前置动作提交为 completed，本动作才允许开始」；同计划内相互依赖用局部
``key`` 引用，程序会解析为真实身份。禁止自由文本动作、``do:[散文]``
或自定义 operation；已有动作重声明必须携带真实 id 与完整参数。
"""

from sceneweave.domain.action_plan import ActionPlan
from sceneweave.domain.identity import ResponseId
from sceneweave.domain.tracing import TurnTracer
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import OwnActionDigest, TurnFacts, trace_stage
from sceneweave.roles.model_output import (
    ModelOutputError,
    System3Unfulfillable,
    parse_system3_output,
)

SCHEMA = "[动作对象, 动作对象, ...]（一维数组；没有主动行动时输出 []）"

# 独立的意向不可落实控制对象：不是动作，也不是 World 客观事实。
FAILURE_SCHEMA = '{"error": "intention_unfulfillable", "reason": "具体原因"}'

# 动作字段表：固定 schema，模型不得自造种类或字段。
_ACTION_TABLE = "\n".join(
    [
        '{"kind":"reach","target":"实体标识","effector":"left_hand|right_hand"} 伸手触及',
        '{"kind":"grasp","target":"实体标识","effector":"left_hand|right_hand"} 抓取',
        '{"kind":"release","target":"实体标识","effector":"left_hand|right_hand"} 松开',
        '{"kind":"move","destination":{"entity":"实体标识"}或{"position":{"x":数值,"y":数值,"z":数值可选}},'
        '"mode":"walk|brisk|slow|run|sneak"（可选，默认 walk）} 移动',
        '{"kind":"say","content":"台词原文","target":"实体标识（可选）"} 说话（台词允许文本）',
        '{"kind":"look","target":"实体标识（可选）"} 看',
        '{"kind":"wait"} 等待',
        '{"kind":"stand"} 站起',
        '{"kind":"sit"} 坐下',
        '{"kind":"give","target":"接收者实体标识","item":"物品实体标识","effector":"left_hand|right_hand"} 交付',
    ]
)


def build_prompt(
    facts: TurnFacts,
    intention: str,
    own_actions: tuple[OwnActionDigest, ...],
    context_note: str | None,
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
        f"当前剧情时间 {facts.simulation_time} 毫秒。想达成的意向：{intention}",
        "私人认知：",
        *([f"- {item}" for item in memory] or ["- 无"]),
        "本轮感知（原始已知观测）：",
        *observations,
        "自身行动状态（引擎确认的细节）：",
        *known,
        "自身仍在主动维持的行动（重声明时必须连同真实 id 与完整参数给出）：",
        *actions,
    ]
    if context_note is not None:
        lines.append(f"需要针对的问题：{context_note}")
    if attempt > 1:
        lines.append(f"这是第 {attempt} 次落实尝试。")
    lines += [
        "第三步：给出从现在起你要主动实施或维持的完整动作，用一维数组表达。",
        "只允许下面这些结构化原子动作，字段固定；禁止用字符串动作、自由散文、"
        "自定义 operation，也不要把“走过去并拿走某物”塞成一个动作——该拆分就拆成多个动作"
        "（伸手与抓取分别是两个动作）。",
        _ACTION_TABLE,
        "已有行动要延续：动作对象里带上它的真实 id（从上方行动列表抄），并给出完整参数，"
        "身份与已发生的进展、开始时间保持不变，不要重新计时。",
        "同 ID 必须保持相同动作参数与依赖；改变做法时，从完整链省略旧身份并新增动作，"
        "不要输出修改指令。",
        "新动作不要写 id，由程序分配真实身份。",
        "若某个新动作必须等另一个动作真正完成后才能开始，用 depends_on 列出它依赖的动作："
        "同计划内新动作可用它自己的 key 互相引用（给被依赖的新动作写一个 key，"
        "在依赖方写 depends_on:[那个 key]）；依赖一项已完成的历史动作就写它的真实 id。"
        "depends_on 是可选的，没有依赖就不写。",
        "依赖表示“前置动作被 World 实际提交为完成”，不是预计结束、也不是到点就成功；"
        "多个依赖要全部完成。",
        "不再主动维持某个行动，就从完整计划里省略它（省略不等于世界中的效果消失）。",
        "没有主动行动时输出 []；[] 表示不再主动维持任何行为，不是格式错误。",
        "动作区间与先后由后续翻译器按客观条件安排；你只决定要做什么、依赖什么。",
        "只有当你依据当前已知条件确实找不到任何能满足意向的动作方案时，才改为输出这个"
        "独立的控制对象（它不是动作，也不是世界事实）：",
        FAILURE_SCHEMA,
        "一般的可执行动作、暂时不确定成功与否、或只是被检查打回，都不要用它；"
        "它只表示“在保持当前意向的前提下，当前已知条件下无法落实”。",
        "只输出这个 JSON 一维数组本身（或上面的控制对象），不要任何解释，不要包装字段：",
        SCHEMA,
    ]
    return "\n".join(lines)


def render_plan(plan: ActionPlan) -> str:
    """计划的稳定 display text（依赖以原始引用展示）。"""
    return plan.render()


async def run(
    sampler: TextSampler,
    facts: TurnFacts,
    intention: str,
    own_actions: tuple[OwnActionDigest, ...],
    *,
    response_id: ResponseId,
    context_note: str | None = None,
    attempt: int = 1,
    background: str = "",
    memory: tuple[str, ...] = (),
    tracer: TurnTracer | None = None,
) -> ActionPlan | System3Unfulfillable:
    """执行 System3；解析失败如实传播，不转换为空计划。"""
    prompt = build_prompt(facts, intention, own_actions, context_note, attempt, background, memory)
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
        outcome = parse_system3_output(raw)
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
    if isinstance(outcome, System3Unfulfillable):
        trace_stage(
            tracer,
            "system3.unfulfillable",
            facts,
            response_id,
            "System3 判断当前已知条件无法落实意向（独立失败信号）",
            prompt=prompt,
            raw_output=raw,
            details={"reason": outcome.reason, "attempt": attempt},
        )
        return outcome
    trace_stage(
        tracer,
        "system3.declare",
        facts,
        response_id,
        "System3 给出完整结构化动作计划",
        prompt=prompt,
        raw_output=raw,
        details={
            "actions": len(outcome.actions),
            "with_dependencies": sum(1 for item in outcome.actions if item.depends_on),
            "plan": render_plan(outcome),
            "attempt": attempt,
        },
    )
    return outcome
