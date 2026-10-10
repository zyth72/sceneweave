"""从本地 AISHELL-3 压缩包提取语速参考，并用留出的测试录音评价耗时基线。

流式读取归档，不解压整套音频，不调用模型。只用 train 建立参考，test 用于评价。
首尾裁剪使用振幅门限，保留中间停顿；其跨度是启发式估计，不是 VAD 真值。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tarfile
import wave
from array import array
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

WINDOW_MS = 20
PADDING_MS = 40
RMS_THRESHOLD = 100
METRICS = ("file_duration_ms", "estimated_trimmed_span_ms")


class DataError(ValueError):
    """数据格式或配对错误。"""


@dataclass(frozen=True)
class Transcript:
    filename: str
    words: tuple[str, ...]
    pinyin: tuple[str, ...]

    @property
    def speaker(self) -> str:
        return self.filename[:7]

    @property
    def text(self) -> str:
        return "".join(self.words)


@dataclass
class Sample:
    split: str
    utterance_id: str
    speaker: str
    source_member: str
    text: str
    pinyin: tuple[str, ...]
    unit_count: int
    char_count: int
    metadata: dict[str, str]
    file_duration_ms: float | None = None
    estimated_trimmed_span_ms: float | None = None
    leading_trim_ms: float | None = None
    trailing_trim_ms: float | None = None
    sample_rate: int | None = None
    error: str | None = None
    predictions: dict[str, Any] = field(default_factory=dict)


def parse_transcripts(text: str) -> dict[str, Transcript]:
    result: dict[str, Transcript] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) < 3 or len(tokens) % 2 != 1:
            raise DataError(f"转录第 {number} 行不是文字与拼音交替格式")
        filename = tokens[0]
        if not re.fullmatch(r"SSB\d{8}\.wav", filename) or filename in result:
            raise DataError(f"转录第 {number} 行身份非法或重复")
        words, phones = tuple(tokens[1::2]), tuple(tokens[2::2])
        if any(not re.fullmatch(r"[a-züv]+[1-5]", phone) for phone in phones):
            raise DataError(f"转录第 {number} 行拼音非法")
        if any(not re.fullmatch(r"[\u3400-\u9fff]+", word) for word in words):
            raise DataError(f"转录第 {number} 行文字非法")
        result[filename] = Transcript(filename, words, phones)
    return result


def select_ids(transcripts: dict[str, Transcript], limit: int, seed: int, split: str) -> set[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for filename, item in transcripts.items():
        grouped[item.speaker].append(filename)
    selected: set[str] = set()
    for ids in grouped.values():
        ids.sort(key=lambda name: hashlib.sha256(f"{seed}/{split}/{name}".encode()).digest())
        selected.update(ids[:limit])
    return selected


def read_metadata(text: str) -> tuple[dict[str, dict[str, str]], str]:
    people: dict[str, dict[str, str]] = {}
    header: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            header.append(line)
        elif line.strip():
            tokens = line.split()
            if len(tokens) != 4 or tokens[0] in people:
                raise DataError("说话人元数据非法或重复")
            people[tokens[0]] = dict(
                zip(("age_group", "gender", "accent"), tokens[1:], strict=True)
            )
    return people, "\n".join(header)


def trim_bounds(pcm: Sequence[int], sample_rate: int) -> tuple[int, int]:
    window = max(1, round(sample_rate * WINDOW_MS / 1000))
    padding = round(sample_rate * PADDING_MS / 1000)

    def audible(start: int, end: int) -> bool:
        chunk = pcm[start:end]
        return math.sqrt(sum(value * value for value in chunk) / len(chunk)) >= RMS_THRESHOLD

    left = 0
    while left < len(pcm) and not audible(left, min(left + window, len(pcm))):
        left += window
    if left >= len(pcm):
        raise DataError("全静音或音量未达到固定振幅门限")
    right = len(pcm)
    while right > left and not audible(max(left, right - window), right):
        right -= window
    return max(0, left - padding), min(len(pcm), right + padding)


def measure_wav(payload: bytes) -> dict[str, float | int]:
    with wave.open(io.BytesIO(payload), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise DataError("仅支持单声道 16bit PCM")
        rate, frames = audio.getframerate(), audio.getnframes()
        if rate <= 0 or frames <= 0:
            raise DataError("音频为空或采样率非法")
        raw = audio.readframes(frames)
        if len(raw) != frames * 2:
            raise DataError("WAV 数据长度与头部不符")
    pcm = array("h")
    pcm.frombytes(raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    left, right = trim_bounds(pcm, rate)
    return {
        "file_duration_ms": frames * 1000 / rate,
        "estimated_trimmed_span_ms": (right - left) * 1000 / rate,
        "leading_trim_ms": left * 1000 / rate,
        "trailing_trim_ms": (frames - right) * 1000 / rate,
        "sample_rate": rate,
    }


def scan_archive(
    archive_path: Path, samples: list[Sample], train_limit: int, test_limit: int, seed: int
) -> dict[str, Any]:
    transcripts: dict[str, dict[str, Transcript]] = {}
    selected: dict[str, set[str]] = {}
    found: dict[str, set[str]] = defaultdict(set)
    people: dict[str, dict[str, str]] = {}
    header = ""
    limits = {"train": train_limit, "test": test_limit}
    audio_count: Counter[str] = Counter()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            if member.name == "spk-info.txt" or member.name in (
                "train/content.txt",
                "test/content.txt",
            ):
                handle = archive.extractfile(member)
                assert handle is not None
                text = handle.read().decode("utf-8-sig")
                if member.name == "spk-info.txt":
                    people, header = read_metadata(text)
                else:
                    split = name.parts[0]
                    if split in transcripts:
                        raise DataError(f"重复的 {member.name}")
                    transcripts[split] = parse_transcripts(text)
                    selected[split] = select_ids(transcripts[split], limits[split], seed, split)
                continue
            parts = name.parts
            if (
                len(parts) != 4
                or parts[0] not in limits
                or parts[1] != "wav"
                or name.suffix != ".wav"
            ):
                continue
            split, filename = parts[0], parts[3]
            if split not in transcripts:
                raise DataError("归档顺序需先有 content.txt，再有该 split 的 WAV")
            if filename in found[split]:
                raise DataError(f"重复音频：{member.name}")
            found[split].add(filename)
            audio_count[split] += 1
            if sum(audio_count.values()) % 20000 == 0:
                print(f"已扫描 {sum(audio_count.values())} 条音频", flush=True)
            if filename not in selected[split]:
                continue
            item = transcripts[split][filename]
            sample = Sample(
                split,
                filename,
                item.speaker,
                member.name,
                item.text,
                item.pinyin,
                len(item.words),
                len(item.text),
                people.get(item.speaker, {}),
            )
            samples.append(sample)
            try:
                if parts[2] != item.speaker or member.size > 64 * 1024 * 1024:
                    raise DataError("音频身份不符或单文件超过 64MiB")
                handle = archive.extractfile(member)
                assert handle is not None
                measurements = measure_wav(handle.read())
                sample.file_duration_ms = float(measurements["file_duration_ms"])
                sample.estimated_trimmed_span_ms = float(measurements["estimated_trimmed_span_ms"])
                sample.leading_trim_ms = float(measurements["leading_trim_ms"])
                sample.trailing_trim_ms = float(measurements["trailing_trim_ms"])
                sample.sample_rate = int(measurements["sample_rate"])
            except (DataError, wave.Error, EOFError) as error:
                sample.error = str(error)
    if set(transcripts) != set(limits) or not people:
        raise DataError("缺少 train/test 转录本或说话人元数据")
    pairing: dict[str, Any] = {}
    for split in limits:
        missing = set(transcripts[split]) - found[split]
        pairing[split] = {
            "transcript_count": len(transcripts[split]),
            "audio_count": audio_count[split],
            "missing_audio": len(missing),
            "unmatched_audio": len(found[split] - set(transcripts[split])),
            "selected_count": len(selected[split]),
        }
        for filename in sorted(missing & selected[split]):
            item = transcripts[split][filename]
            samples.append(
                Sample(
                    split,
                    filename,
                    item.speaker,
                    f"{split}/wav/{item.speaker}/{filename}",
                    item.text,
                    item.pinyin,
                    len(item.words),
                    len(item.text),
                    people.get(item.speaker, {}),
                    error="转录本对应音频缺失",
                )
            )
    speakers = {
        split: {item.speaker for item in items.values()} for split, items in transcripts.items()
    }
    return {
        "pairing": pairing,
        "metadata_header": header,
        "corpus_speakers": {split: len(ids) for split, ids in speakers.items()},
        "corpus_speaker_overlap": len(speakers["train"] & speakers["test"]),
        "corpus_test_unseen_speakers": len(speakers["test"] - speakers["train"]),
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
    }


def duration(sample: Sample, metric: str) -> float:
    value = getattr(sample, metric)
    if value is None or value <= 0:
        raise DataError("参与评价的音频时长必须为正")
    return float(value)


def error_metrics(errors: list[tuple[float, float]]) -> dict[str, Any]:
    if not errors:
        return {
            "count": 0,
            "mae_ms": None,
            "mape": None,
            "median_relative_error": None,
            "within_20_percent": None,
        }
    absolute = [abs(predicted - actual) for predicted, actual in errors]
    relative = [abs(predicted - actual) / actual for predicted, actual in errors]
    return {
        "count": len(errors),
        "mae_ms": statistics.mean(absolute),
        "mape": statistics.mean(relative),
        "median_relative_error": statistics.median(relative),
        "within_20_percent": sum(item <= 0.2 for item in relative) / len(relative),
    }


def analyze(samples: list[Sample]) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [item for item in samples if item.split == "train" and item.error is None]
    test = [item for item in samples if item.split == "test" and item.error is None]
    if not train:
        raise DataError("没有有效训练样本")
    people = {item.speaker for item in train}
    profiles: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    for metric in METRICS:
        grouped: dict[str, list[float]] = defaultdict(list)
        for item in train:
            grouped[item.speaker].append(item.unit_count * 1000 / duration(item, metric))
        global_profile = distribution([rate for values in grouped.values() for rate in values])
        speaker_profiles = {
            person: distribution(values) for person, values in sorted(grouped.items())
        }
        profiles[metric] = {"global_train": global_profile, "speakers": speaker_profiles}
        buckets: dict[str, dict[str, list[tuple[float, float]]]] = {
            group: {method: [] for method in ("global_train", "speaker_train_fallback")}
            for group in ("all", "seen", "unseen")
        }
        for item in test:
            actual = duration(item, metric)
            known = item.speaker in speaker_profiles
            global_rate = global_profile["median"]
            personal_rate = speaker_profiles[item.speaker]["median"] if known else global_rate
            predictions = {
                "global_train": item.unit_count * 1000 / global_rate,
                "speaker_train_fallback": item.unit_count * 1000 / personal_rate,
            }
            item.predictions[metric] = {
                **predictions,
                "reference": "same_speaker_train" if known else "global_train_fallback",
            }
            for method, predicted in predictions.items():
                buckets["all"][method].append((predicted, actual))
                buckets["seen" if known else "unseen"][method].append((predicted, actual))
        evaluation[metric] = {
            group: {method: error_metrics(pairs) for method, pairs in methods.items()}
            for group, methods in buckets.items()
        }
    return profiles, {
        "sample_count": len(samples),
        "errors": sum(item.error is not None for item in samples),
        "valid_by_split": {"train": len(train), "test": len(test)},
        "sampled_speakers": {"train": len(people), "test": len({item.speaker for item in test})},
        "test_seen_count": sum(item.speaker in people for item in test),
        "test_unseen_count": sum(item.speaker not in people for item in test),
        "evaluation": evaluation,
    }


def render_report(samples: list[Sample], profiles: dict[str, Any], summary: dict[str, Any]) -> str:
    def number(value: Any) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [
        "# AISHELL-3 说话耗时辅助数据分析",
        "",
        f"有效样本：train={summary['valid_by_split']['train']}、test={summary['valid_by_split']['test']}；"
        f"错误样本 {summary['errors']}。原始数据盘点见 summary.json 的 inventory。",
        f"测试样本中，已见说话人 {summary['test_seen_count']} 条，未见说话人 {summary['test_unseen_count']} 条。",
        "",
        "## 口径",
        "",
        "- 这是每人限量的确定性分层抽样，结果不代表原始数据的频率分布，也不是全量 88,035 条耗时分析。",
        "- unit_count 是文字/拼音标注对数；char_count 是文字字符数。儿化词可能多字一音，不能混用。",
        "- file_duration_ms 保留整条录音；estimated_trimmed_span_ms 用 20ms RMS 窗口、阈值100、40ms边缘padding估计。",
        "- 裁剪只删首尾留白，保留内部停顿；固定振幅门限不是语音识别/VAD真值，低音量及背景噪声会影响它。",
        "- 语速参考只从 train 形成；test 真实耗时仅作评价。已知说话人用其 train 中位语速，未知说话人回退总体 train 中位语速。",
        "- 各说话人的 p10/p90 是训练样本经验分位数，不是已校准的预测置信区间。",
        "",
        "## 训练语速参考（标注单元/秒）",
        "",
        "| 时长口径 | 样本 | p10 | 中位 | p90 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        data = profiles[metric]["global_train"]
        lines.append(
            f"| {metric} | {data['count']} | {number(data['p10'])} | {number(data['median'])} | {number(data['p90'])} |"
        )
    lines += [
        "",
        "## 留出测试预测误差",
        "",
        "预测耗时=标注单元数/训练语速。下列分开报告已见/未见说话人，避免把同人留出当作新角色泛化。",
        "",
        "| 时长口径 | 分组 | 方法 | n | MAE(ms) | MAPE(%) | 中位相对误差(%) | ±20%覆盖(%) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        for group, methods in summary["evaluation"][metric].items():
            for method, data in methods.items():
                percentages = [
                    None if data[key] is None else data[key] * 100
                    for key in ("mape", "median_relative_error", "within_20_percent")
                ]
                lines.append(
                    f"| {metric} | {group} | {method} | {data['count']} | {number(data['mae_ms'])} | "
                    + " | ".join(number(value) for value in percentages)
                    + " |"
                )
    lines += [
        "",
        "## 逐条示例",
        "",
        "完整样本和两种时长口径的预测见 samples.jsonl；下表只展示部分测试样本，耗时单位ms。",
        "",
        "| 来源 | 说话人 | 文本 | 单元/字 | 录音总长 | 裁剪估计 | 裁剪口径个人参考预测 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in [sample for sample in samples if sample.split == "test"][:30]:
        prediction = item.predictions.get("estimated_trimmed_span_ms", {}).get(
            "speaker_train_fallback"
        )
        lines.append(
            f"| {item.source_member} | {item.speaker} | {item.text} | {item.unit_count}/{item.char_count} | "
            f"{number(item.file_duration_ms)} | {number(item.estimated_trimmed_span_ms)} | {number(prediction)} |"
        )
    lines += [
        "",
        "## 如何用于翻译器",
        "",
        "1. 把训练语速分布及少量训练例子作为说话耗时参考，先做独立的有/无辅助数据探针对照。",
        "2. 给模型的是 say 文本和允许使用的参考条件；测试目标耗时留作评价，不放进模型输入。",
        "3. 先比较模型与上述简单基线的误差，再决定是否值得微调；这次没有运行模型对照或微调。",
        "- 这是中性朗读语料，没有疲劳、紧张、争吵的对应标签，不能从元数据推导角色的因果修正系数。",
        "- 音频帮助估计说话执行跨度，不能决定行动开始延迟、World成功结果、打断或资源冲突。",
        "- 原始数据保持独立，遵循 AISHELL-3 的 Apache 2.0 许可，不随代码仓库提交。",
        "",
        "[数据源](https://openslr.org/93/) · [样本](samples.jsonl) · [训练参考](speaker_profiles.json) · [配置](manifest.json)",
    ]
    return "\n".join(lines) + "\n"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--train-per-speaker", type=int, default=16)
    parser.add_argument("--test-per-speaker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.train_per_speaker, args.test_per_speaker) <= 0:
        parser.error("每位说话人的样本数必须为正整数")
    archive, outdir = args.archive.expanduser().resolve(), args.outdir.expanduser().resolve()
    stat = archive.stat()
    if outdir.exists() and any(outdir.iterdir()):
        parser.error("输出目录非空，拒绝覆写")
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "created_utc": datetime.now(UTC).isoformat(),
        "archive": str(archive),
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {
            "train_per_speaker": args.train_per_speaker,
            "test_per_speaker": args.test_per_speaker,
            "seed": args.seed,
            "window_ms": WINDOW_MS,
            "padding_ms": PADDING_MS,
            "rms_threshold": RMS_THRESHOLD,
        },
        "source": "https://openslr.org/93/",
        "license": "Apache-2.0",
    }
    write_json(outdir / "manifest.json", manifest)
    samples: list[Sample] = []
    try:
        inventory = scan_archive(
            archive, samples, args.train_per_speaker, args.test_per_speaker, args.seed
        )
        profiles, summary = analyze(samples)
        summary["inventory"] = inventory
        write_json(outdir / "speaker_profiles.json", profiles)
        write_json(outdir / "summary.json", summary)
        (outdir / "report.md").write_text(
            render_report(samples, profiles, summary), encoding="utf-8"
        )
        bad_pairing = any(
            row["missing_audio"] or row["unmatched_audio"] for row in inventory["pairing"].values()
        )
        manifest["status"] = (
            "complete_with_errors" if summary["errors"] or bad_pairing else "complete"
        )
        manifest["metadata_header"] = inventory["metadata_header"]
    except (DataError, OSError, tarfile.TarError, EOFError, UnicodeError) as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        print(f"分析失败：{error}", file=sys.stderr)
    finally:
        (outdir / "samples.jsonl").write_text(
            "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in samples),
            encoding="utf-8",
        )
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(outdir / "manifest.json", manifest)
    print(f"状态：{manifest['status']}；报告：{outdir / 'report.md'}", flush=True)
    return int(manifest["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
