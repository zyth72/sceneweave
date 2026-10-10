"""用微型归档和合成音频验证辅助数据口径，不访问真实数据或模型。"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import wave
from array import array
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "aishell_analysis_under_test", Path(__file__).parents[1] / "scripts/analyze_aishell3.py"
)
assert _SPEC and _SPEC.loader
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def wav_bytes(pcm: list[int], rate: int = 1000) -> bytes:
    buffer = io.BytesIO()
    frames = array("h", pcm)
    if sys.byteorder != "little":
        frames.byteswap()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return buffer.getvalue()


def make_archive(path: Path, test_frames: int = 2000) -> None:
    members: dict[str, bytes] = {
        "spk-info.txt": b"# age groups as provided\nSSB0001 B female north\nSSB0002 C male south\nSSB0003 D male north\n",
        "train/content.txt": "SSB00010001.wav 你 ni3 好 hao3\nSSB00020001.wav 今 jin1 天 tian1 天 tian1 气 qi4\n".encode(),
        "train/wav/SSB0001/SSB00010001.wav": wav_bytes([3000] * 1000),
        "train/wav/SSB0002/SSB00020001.wav": wav_bytes([3000] * 1000),
        "test/content.txt": "SSB00010002.wav 今 jin1 天 tian1 天 tian1 气 qi4\nSSB00030001.wav 今 jin1 天 tian1 天 tian1 气 qi4\n".encode(),
        "test/wav/SSB0001/SSB00010002.wav": wav_bytes([3000] * test_frames),
        "test/wav/SSB0003/SSB00030001.wav": wav_bytes([3000] * 2000),
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_transcript_erhua_keeps_units_separate_from_characters() -> None:
    parsed = analysis.parse_transcripts("SSB00010001.wav\t今儿 jinr1 好 hao3")
    item = parsed["SSB00010001.wav"]
    assert item.text == "今儿好"
    assert len(item.words) == 2 and len(item.text) == 3
    assert item.pinyin == ("jinr1", "hao3")
    for bad in ("SSB00010001.wav 你 ni3 好", "../outside.wav 你 ni3", "SSB00010001.wav 你 wrong"):
        with pytest.raises(analysis.DataError):
            analysis.parse_transcripts(bad)


def test_trim_keeps_internal_pause_and_reports_original_duration() -> None:
    # 200ms留白 + 100ms声音 + 400ms句中停顿 + 100ms声音 + 200ms留白。
    payload = wav_bytes([0] * 200 + [3000] * 100 + [0] * 400 + [3000] * 100 + [0] * 200)
    measured = analysis.measure_wav(payload)
    assert measured["file_duration_ms"] == 1000
    assert measured["leading_trim_ms"] == measured["trailing_trim_ms"] == 160
    assert measured["estimated_trimmed_span_ms"] == 680
    with pytest.raises(analysis.DataError, match="全静音"):
        analysis.measure_wav(wav_bytes([0] * 1000))


def test_small_archive_profiles_exclude_test_targets_and_mark_fallback(tmp_path: Path) -> None:
    first, changed = tmp_path / "first.tgz", tmp_path / "changed.tgz"
    make_archive(first)
    make_archive(changed, test_frames=4000)
    runs = []
    for archive in (first, changed):
        samples = []
        inventory = analysis.scan_archive(archive, samples, 16, 8, 0)
        profiles, summary = analysis.analyze(samples)
        runs.append((samples, profiles, summary))
        assert inventory["corpus_speaker_overlap"] == 1
        assert inventory["corpus_test_unseen_speakers"] == 1
        assert all(
            row["missing_audio"] == row["unmatched_audio"] == 0
            for row in inventory["pairing"].values()
        )
    assert runs[0][1] == runs[1][1], "改变测试耗时不能改变任何训练语速参考"
    for samples, _profiles, summary in runs:
        known = next(item for item in samples if item.utterance_id == "SSB00010002.wav")
        unknown = next(item for item in samples if item.utterance_id == "SSB00030001.wav")
        prediction = known.predictions["file_duration_ms"]
        assert prediction["speaker_train_fallback"] == 2000
        assert prediction["global_train"] == pytest.approx(4000 / 3)
        assert unknown.predictions["file_duration_ms"]["reference"] == "global_train_fallback"
        assert summary["test_seen_count"] == summary["test_unseen_count"] == 1


def test_cli_outputs_traceable_samples_and_refuses_overwrite(tmp_path: Path) -> None:
    source, outdir = tmp_path / "source.tgz", tmp_path / "result"
    make_archive(source)
    args = ["--archive", str(source), "--outdir", str(outdir)]
    assert analysis.main(args) == 0
    manifest = json.loads((outdir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    samples = [json.loads(line) for line in (outdir / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 4 and all(item["source_member"] for item in samples)
    report = (outdir / "report.md").read_text()
    assert "今天天气" in report and "test/wav/SSB0001/SSB00010002.wav" in report
    before = (outdir / "samples.jsonl").read_bytes()
    with pytest.raises(SystemExit):
        analysis.main(args)
    assert (outdir / "samples.jsonl").read_bytes() == before


def test_corrupt_archive_is_failed_not_complete(tmp_path: Path) -> None:
    source, outdir = tmp_path / "bad.tgz", tmp_path / "result"
    source.write_bytes(b"not a gzip archive")
    assert analysis.main(["--archive", str(source), "--outdir", str(outdir)]) == 1
    manifest = json.loads((outdir / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"]
