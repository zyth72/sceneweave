"""统一运行入口：uv run --locked python -m sceneweave <config> [mode]。

三种模式：
  --live module:callable [options]   真实 LanguageModel（用户注入）
  --scripted script.json             脚本假模型（不调真实 API）
  --replay record_dir                重放录制目录（不导入 live 工厂）

--record DIR 与 --replay 互斥。--ui [PORT] 启动真实 SimulationServer。

--live 或显式 --audit-config 用 python-dotenv 加载当前工作目录 .env
（不向上搜索；终端已有变量优先；不插值）。未开启审计的 scripted、
replay 与 --help 不读取 .env，程序化工厂仍只使用已传入的环境变量。
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from sceneweave.app.assembly import (
    AssemblyError,
    build_live_factory,
    build_scripted_factory,
)
from sceneweave.app.recording import RecordedRun
from sceneweave.app.runner import RunnerError, RunSummary, run_headless
from sceneweave.audit.config import AuditModelSpec, load_audit_config
from sceneweave.audit.session import AuditSession
from sceneweave.config.loader import ConfigError, load_assembly_config, parse_json_object
from sceneweave.domain.modeling import TextSampler
from sceneweave.integrations.concordia.sampling import ReplayMismatchError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sceneweave", description="sceneweave 统一运行入口")
    parser.add_argument("config", type=Path, help="装配配置 JSON 文件路径")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", metavar="MODULE:CALLABLE", help="真实模型工厂")
    mode.add_argument("--scripted", metavar="SCRIPT_JSON", type=Path, help="脚本假模型 JSON")
    mode.add_argument("--replay", metavar="RECORD_DIR", type=Path, help="重放录制目录")
    parser.add_argument("--live-options", default="{}", help="live 工厂 JSON options")
    parser.add_argument("--record", metavar="DIR", type=Path, help="录制目录")
    parser.add_argument("--output", metavar="DIR", type=Path, default=None, help="输出目录")
    parser.add_argument("--ui", nargs="?", const=8080, type=int, metavar="PORT", help="启动 UI")
    parser.add_argument(
        "--audit-config",
        metavar="PATH",
        type=Path,
        default=None,
        help="运行后审计配置 JSON；缺省不审计、不读 .env",
    )
    return parser


def _load_live_dotenv() -> None:
    """真实模拟或审计加载当前工作目录 .env（显式路径，不向上搜索）。

    只调用 python-dotenv 官方加载器：override=False 保留终端已有变量，
    interpolate=False 不解析变量引用；无 .env 文件正常沿用环境变量。
    读取失败只报异常类别，不回显文件内容或任何值。
    """
    from dotenv import load_dotenv

    try:
        load_dotenv(
            dotenv_path=Path.cwd() / ".env",
            override=False,
            interpolate=False,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError(f"当前工作目录 .env 加载失败（{type(error).__name__}）") from None


def _load_scripted(path: Path) -> Any:
    from sceneweave.config.loader import load_json_object

    try:
        data = load_json_object(path)
    except (ConfigError, OSError) as error:
        raise RunnerError(f"脚本文件无法加载：{error}") from error
    for scope, components in data.items():
        if not isinstance(scope, str) or not scope:
            raise RunnerError(f"脚本作用域必须是非空字符串：{scope!r}")
        if not isinstance(components, dict):
            raise RunnerError(f"脚本作用域 {scope}：component 映射必须是对象")
        for component, responses in components.items():
            if component not in {
                "system1",
                "system2",
                "system3",
                "translator",
                "world-check",
                "world-progress",
                "world-adjudicate",
            }:
                raise RunnerError(f"脚本 {scope}.{component}：未知组件名")
            if not isinstance(responses, list) or not all(isinstance(r, str) for r in responses):
                raise RunnerError(f"脚本 {scope}.{component}：回答队列必须是字符串数组")
    return build_scripted_factory(data)


def _default_audit_factory(spec: AuditModelSpec) -> TextSampler:
    """默认审计模型工厂：复用已有 live 工厂（构造不读环境、不联网）。"""
    factory = build_live_factory(spec.factory, spec.options)
    return factory(f"audit:{spec.model_id}")


def _load_audit(args: Any) -> AuditSession | None:
    """审计显式开启：严格配置 + 预构造模型；此时才读当前目录 .env。"""
    if args.audit_config is None:
        return None
    audit_config = load_audit_config(args.audit_config)
    if args.live is None:
        _load_live_dotenv()
    session = AuditSession(audit_config, _default_audit_factory)
    session.prepare()
    return session


def _print_audit(session: AuditSession | None) -> int:
    if session is None:
        return 0
    state = session.state.snapshot()
    error = state.get("error")
    suffix = f"（{error}）" if error else ""
    print(f"审计状态：{state['status']}{suffix}")
    for model in state["models"]:
        detail = model.get("error") or model.get("summary") or ""
        model_suffix = f" — {detail}" if detail else ""
        print(f"  模型 {model['id']}（{model['model']}）：{model['status']}{model_suffix}")
    failed = bool(state["failed_models"]) or state["status"] in ("failed", "cancelled")
    return 1 if failed else 0


def _print_result(summary: RunSummary, out_dir: Path) -> int:
    print(f"停止原因：{summary.stop_reason}")
    print(f"最终剧情时刻：{summary.final_time_ms}ms")
    print(f"完成周期：{summary.completed_cycles}")
    print(f"世界版本：{summary.world_state_version}")
    print(f"输出目录：{out_dir}")
    if summary.error:
        print(f"错误：{summary.error}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.record is not None and args.replay is not None:
        print("错误：--record 与 --replay 互斥", file=sys.stderr)
        return 2

    if args.ui is not None:
        from sceneweave.app.ui import UIRunnerError, _validate_port

        try:
            _validate_port(args.ui)
        except UIRunnerError as error:
            print(f"端口错误：{error}", file=sys.stderr)
            return 2

    try:
        config = load_assembly_config(args.config)
    except (ConfigError, OSError) as error:
        print(f"配置错误：{error}", file=sys.stderr)
        return 2

    try:
        audit_session = _load_audit(args)
    except (ConfigError, AssemblyError, OSError) as error:
        print(f"审计配置错误：{error}", file=sys.stderr)
        return 2

    sampler_factory: Any
    replay_run: RecordedRun | None = None
    try:
        if args.live is not None:
            options = parse_json_object(args.live_options, "--live-options")
            _load_live_dotenv()
            sampler_factory = build_live_factory(args.live, options)
        elif args.scripted is not None:
            sampler_factory = _load_scripted(args.scripted)
        else:
            if not args.replay.is_dir():
                raise RunnerError(f"重放目录不存在：{args.replay}")
            replay_run = RecordedRun(args.replay, config)
            sampler_factory = replay_run.for_scope
    except (
        AssemblyError,
        RunnerError,
        ConfigError,
        OSError,
        ReplayMismatchError,
    ) as error:
        print(f"模式错误：{error}", file=sys.stderr)
        return 2

    if args.ui is not None:
        from sceneweave.app.ui import UIRunnerError, run_ui

        def _on_server_ready(address: str, actual_port: int) -> None:
            print(f"UI 已启动：{address}（端口 {actual_port}）")

        async def _run() -> tuple[RunSummary, Path]:
            return await run_ui(
                config,
                sampler_factory,
                port=args.ui,
                output_directory=args.output,
                record_directory=args.record,
                replay_run=replay_run,
                wait_after_finish=True,
                on_server_ready=_on_server_ready,
                audit=audit_session,
            )

        try:
            summary, out_dir = asyncio.run(_run())
        except (
            ConfigError,
            RunnerError,
            AssemblyError,
            UIRunnerError,
            ReplayMismatchError,
            OSError,
        ) as error:
            print(f"UI 运行失败：{error}", file=sys.stderr)
            _print_audit(audit_session)
            return 1
        except KeyboardInterrupt:
            print("UI 运行被中止", file=sys.stderr)
            return 130
        sim_code = _print_result(summary, out_dir)
        audit_code = _print_audit(audit_session)
        return sim_code or audit_code

    async def _run_headless() -> tuple[RunSummary, Path]:
        return await run_headless(
            config,
            sampler_factory,
            output_directory=args.output,
            record_directory=args.record,
            replay_run=replay_run,
            audit=audit_session,
        )

    try:
        summary, out_dir = asyncio.run(_run_headless())
    except (ConfigError, RunnerError, AssemblyError, ReplayMismatchError, OSError) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        _print_audit(audit_session)
        return 1
    except KeyboardInterrupt:
        print("运行被中止", file=sys.stderr)
        return 130

    sim_code = _print_result(summary, out_dir)
    audit_code = _print_audit(audit_session)
    return sim_code or audit_code


if __name__ == "__main__":
    raise SystemExit(main())
