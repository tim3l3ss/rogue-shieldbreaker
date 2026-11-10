# tests/test_bench_utils.py
from pathlib import Path

import pytest

from bench_utils import (
    append_row_to_csv,
    find_run_dir,
    get_already_logged_models,
    get_saved_train_time_sec,
)


def test_find_run_dir_returns_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "thesis_runs" / "yolo11n_pea_1280").mkdir(parents=True)
    assert find_run_dir("yolo11n_pea_1280") == Path("thesis_runs") / "yolo11n_pea_1280"


def test_find_run_dir_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = Path("runs") / "detect" / "thesis_runs" / "missing_run"
    assert find_run_dir("missing_run") == expected


def test_train_time_read_from_results_csv(tmp_path):
    (tmp_path / "results.csv").write_text(
        "epoch,   time\n1,  100.5\n2,  3204.0\n", encoding="utf-8"
    )
    assert get_saved_train_time_sec(tmp_path, fallback_hours=0.89) == 3204.0


def test_train_time_fallback(tmp_path):
    assert get_saved_train_time_sec(tmp_path, fallback_hours=0.89) == pytest.approx(3204.0)


def test_append_writes_header_once(tmp_path):
    csv_path = tmp_path / "results.csv"
    append_row_to_csv(csv_path, {"Model": "yolo11n", "FPS": 50.0})
    append_row_to_csv(csv_path, {"Model": "yolo26s", "FPS": 40.0})
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == "Model,FPS"
    assert get_already_logged_models(csv_path) == {"yolo11n", "yolo26s"}


def test_logged_models_missing_file(tmp_path):
    assert get_already_logged_models(tmp_path / "nope.csv") == set()
