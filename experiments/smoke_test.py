from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


_NAS_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_MAP = _NAS_ROOT / "smoke_model_map.json"

_ALGS_IN_ORDER = [
    "AlphaEdit",
    "NAS",
    "MEMIT_rect",
    "MEMIT_seq",
    "MEMIT_prune",
    "MEMIT",
    "ROME",
    "FT",
    "LyapLock",
]
_DATASETS = ["mcf", "zsre"]


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _read_tail(path: Path, *, max_bytes: int = 65536) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return repr(data)


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def _hparams_dir_for_alg(alg_name: str) -> str:
    return "MEMIT" if "MEMIT" in alg_name else alg_name


def _discover_hparams(alg_name: str) -> list[Path]:
    dir_name = _hparams_dir_for_alg(alg_name)
    hdir = _NAS_ROOT / "hparams" / dir_name
    if not hdir.exists():
        return []
    return sorted(hdir.glob("*.json"), key=lambda p: p.name)


def _infer_model_key(hparams_path: Path, model_map: dict[str, str]) -> Optional[str]:
    try:
        data = _load_json(hparams_path)
    except Exception:
        return None

    if isinstance(data, dict):
        model_name = data.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

    fname = hparams_path.name
    for candidate in sorted(model_map.keys(), key=len, reverse=True):
        if candidate in fname:
            return candidate
    return None


def _resolve_model_name(model_key: Optional[str], model_map: dict[str, str]) -> Optional[str]:
    if not model_key:
        return None
    mapped = model_map.get(model_key)
    if isinstance(mapped, str) and mapped.strip():
        return mapped.strip()
    # Allow hparams to specify the full HF repo id/path directly (e.g., "Qwen/Qwen2.5-7B-Instruct").
    if model_key in model_map.values():
        return model_key
    if "/" in model_key:
        return model_key
    return None


@dataclass
class _CaseResult:
    alg_name: str
    hparams_dir: str
    hparams_fname: str
    ds_name: str
    model_key: Optional[str]
    model_name: Optional[str]
    status: str
    return_code: Optional[int]
    duration_sec: float
    run_dir: str
    error_tail: Optional[str] = None


def _case_key(alg_name: str, hparams_fname: str, ds_name: str) -> tuple[str, str, str]:
    return (alg_name, hparams_fname, ds_name)


def _run_one_case(
    *,
    alg_name: str,
    hparams_path: Path,
    ds_name: str,
    model_map: dict[str, str],
    out_root: Path,
) -> _CaseResult:
    hparams_dir = _hparams_dir_for_alg(alg_name)
    hparams_fname = hparams_path.name

    model_key = _infer_model_key(hparams_path, model_map)
    model_name = _resolve_model_name(model_key, model_map)

    case_dir = out_root / alg_name / hparams_path.stem / ds_name
    run_dir = case_dir / "run"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    cmd_path = case_dir / "cmd.txt"

    cmd = [
        sys.executable,
        "-m",
        "experiments.evaluate",
        "--alg_name",
        alg_name,
        "--model_name",
        model_name or "<MISSING_MODEL_MAP>",
        "--hparams_fname",
        hparams_fname,
        "--ds_name",
        ds_name,
        "--dataset_size_limit",
        "1",
        "--skip_generation_tests",
        "--generation_test_interval",
        "-1",
        "--run_dir_override",
        str(run_dir),
    ]
    cmd_path.write_text(_shell_join(cmd) + "\n", encoding="utf-8")

    if model_name is None:
        return _CaseResult(
            alg_name=alg_name,
            hparams_dir=hparams_dir,
            hparams_fname=hparams_fname,
            ds_name=ds_name,
            model_key=model_key,
            model_name=model_name,
            status="fail",
            return_code=None,
            duration_sec=0.0,
            run_dir=str(run_dir),
            error_tail=(
                f"Missing model mapping for model_key={model_key!r}. "
                f"Update {_DEFAULT_MODEL_MAP}."
            ),
        )

    start = time.time()
    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        proc = subprocess.run(cmd, cwd=_NAS_ROOT, stdout=out_f, stderr=err_f, text=True)
    duration = time.time() - start

    status = "success" if proc.returncode == 0 else "fail"
    error_tail = None
    if status != "success":
        error_tail = _read_tail(stderr_path)

    return _CaseResult(
        alg_name=alg_name,
        hparams_dir=hparams_dir,
        hparams_fname=hparams_fname,
        ds_name=ds_name,
        model_key=model_key,
        model_name=model_name,
        status=status,
        return_code=proc.returncode,
        duration_sec=duration,
        run_dir=str(run_dir),
        error_tail=error_tail,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential smoke test runner for NAS/experiments/evaluate.py. "
            "Runs each (alg, hparams) over mcf and zsre with dataset_size_limit=1, "
            "and records success/failure + logs per case."
        )
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output directory for smoke reports. Defaults to smoke_reports/<timestamp> under NAS.",
    )
    parser.add_argument(
        "--model_map",
        type=str,
        default=str(_DEFAULT_MODEL_MAP),
        help="JSON mapping from hparams model_name / filename key to HF model id/path.",
    )
    args = parser.parse_args()

    model_map_path = Path(args.model_map).expanduser()
    if not model_map_path.is_absolute():
        model_map_path = (_NAS_ROOT / model_map_path).resolve()
    model_map_raw = _load_json(model_map_path)
    if not isinstance(model_map_raw, dict):
        raise TypeError(f"Expected dict in model map JSON: {model_map_path}")
    model_map = {str(k): str(v) for k, v in model_map_raw.items()}

    if args.out_root:
        out_root = Path(args.out_root).expanduser()
        if not out_root.is_absolute():
            out_root = (_NAS_ROOT / out_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = (_NAS_ROOT / "smoke_reports" / stamp).resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "summary.json"
    existing_cases: dict[tuple[str, str, str], dict[str, Any]] = {}
    skip_successes: set[tuple[str, str, str]] = set()
    if summary_path.exists():
        existing_summary = _load_json(summary_path)
        if isinstance(existing_summary, dict):
            existing_list = existing_summary.get("cases", [])
        else:
            existing_list = []

        if isinstance(existing_list, list):
            for item in existing_list:
                if not isinstance(item, dict):
                    continue
                alg = item.get("alg_name")
                hp = item.get("hparams_fname")
                ds = item.get("ds_name")
                if not (isinstance(alg, str) and isinstance(hp, str) and isinstance(ds, str)):
                    continue
                key = _case_key(alg, hp, ds)
                existing_cases[key] = item
                if item.get("status") == "success":
                    skip_successes.add(key)

    active_case_keys: set[tuple[str, str, str]] = set()
    for alg_name in _ALGS_IN_ORDER:
        for hp in _discover_hparams(alg_name):
            for ds_name in _DATASETS:
                active_case_keys.add(_case_key(alg_name, hp.name, ds_name))

    if existing_cases:
        existing_cases = {k: v for k, v in existing_cases.items() if k in active_case_keys}
        skip_successes = {k for k in skip_successes if k in active_case_keys}

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "nas_root": str(_NAS_ROOT),
        "out_root": str(out_root),
        "python": sys.version,
        "model_map": str(model_map_path),
        "algs": list(_ALGS_IN_ORDER),
        "datasets": list(_DATASETS),
    }
    _atomic_write_json(out_root / "meta.json", meta)

    total_cases = 0
    for alg_name in _ALGS_IN_ORDER:
        hparams_files = _discover_hparams(alg_name)
        for hp in hparams_files:
            for ds_name in _DATASETS:
                if _case_key(alg_name, hp.name, ds_name) in skip_successes:
                    continue
                total_cases += 1

    done = 0
    results_by_key: dict[tuple[str, str, str], dict[str, Any]] = dict(existing_cases)
    for alg_name in _ALGS_IN_ORDER:
        hparams_files = _discover_hparams(alg_name)
        if not hparams_files:
            print(f"[WARN] No hparams found for {alg_name} (dir={_hparams_dir_for_alg(alg_name)})")
        for hparams_path in hparams_files:
            for ds_name in _DATASETS:
                key = _case_key(alg_name, hparams_path.name, ds_name)
                if key in skip_successes:
                    print(f"[SKIP] {alg_name} {hparams_path.name} {ds_name} (already success)")
                    continue
                done += 1
                print(f"[{done}/{total_cases}] {alg_name} {hparams_path.name} {ds_name}")
                case = _run_one_case(
                    alg_name=alg_name,
                    hparams_path=hparams_path,
                    ds_name=ds_name,
                    model_map=model_map,
                    out_root=out_root,
                )
                case_dict = asdict(case)
                results_by_key[key] = case_dict

                case_dir = out_root / alg_name / hparams_path.stem / ds_name
                _atomic_write_json(case_dir / "result.json", case_dict)

                ordered = sorted(
                    results_by_key.values(),
                    key=lambda r: (
                        r.get("alg_name", ""),
                        r.get("hparams_fname", ""),
                        r.get("ds_name", ""),
                    ),
                )
                _atomic_write_json(out_root / "summary.json", {"cases": ordered})

    final_cases = list(results_by_key.values())
    successes = sum(1 for r in final_cases if r.get("status") == "success")
    fails = sum(1 for r in final_cases if r.get("status") != "success")
    print(f"Done: {successes} success, {fails} fail. Report: {out_root}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
