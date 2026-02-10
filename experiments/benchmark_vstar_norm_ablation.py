from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_NAS_ROOT = Path(__file__).resolve().parents[1]
if str(_NAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NAS_ROOT))

from baselines.memit import MEMITHyperParams
from dsets import MultiCounterFactDataset
from util.globals import DATA_DIR, HPARAMS_DIR, RESULTS_DIR
from util.vstar_stats import _compute_vstar_norms_batched, _normalize_request


MODELS = [
    {
        "label": "Llama3-8B",
        "hf_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "memit_hparams_path": HPARAMS_DIR / "MEMIT" / "Llama3-8B.json",
    },
    {
        "label": "EleutherAI_gpt-j-6B",
        "hf_model_name": "EleutherAI/gpt-j-6b",
        "memit_hparams_path": HPARAMS_DIR / "MEMIT" / "EleutherAI_gpt-j-6B.json",
    },
]

NS: List[int] = [100, 300, 500, 1000, 2000]
RESTARTS = 5
BASE_SEED = 0
INITIAL_BATCH_SIZE = 50
OUTLIER_UPPER_QUANTILE = 0.95


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or "hip out of memory" in msg
    )


def _recover_from_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _upper_trimmed_mean(
    values: Sequence[float], *, upper_quantile: float
) -> Tuple[float, int, Optional[float]]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot trim empty values")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite v* norms encountered during NAS statistics")

    arr.sort()
    q = float(upper_quantile)
    drop_count = int(np.floor((1.0 - q) * arr.size))
    if drop_count <= 0:
        return float(arr.mean()), 0, float(arr[-1])

    kept = arr[: -drop_count]
    if kept.size == 0:
        raise ValueError("Trimmed away all samples; lower outlier trimming or increase N")
    return float(kept.mean()), int(drop_count), float(kept[-1])


def _build_request_pool() -> List[Dict[str, Any]]:
    ds = MultiCounterFactDataset(DATA_DIR)
    pool: List[Dict[str, Any]] = []
    for record in ds:
        rewrites = record.get("requested_rewrite")
        items = rewrites if isinstance(rewrites, list) else [rewrites]
        for rw in items:
            if not isinstance(rw, dict):
                continue
            request = {
                "prompt": rw.get("prompt"),
                "subject": rw.get("subject"),
                "target_new": rw.get("target_new"),
            }
            request = _normalize_request(request)
            if request is not None:
                pool.append(request)
    if not pool:
        raise RuntimeError("No valid CounterFact rewrites found for v* statistics benchmark.")
    return pool


def _sample_requests(
    pool: List[Dict[str, Any]], *, n: int, seed: int
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    if n <= 0:
        raise ValueError("n must be > 0")
    if n >= len(pool):
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        indices = indices[:n]
    else:
        indices = rng.sample(range(len(pool)), n)
    return [pool[i] for i in indices]


def _load_model_and_tok(hf_model_name: str):
    print(f"Loading model/tokenizer: {hf_model_name}")
    model = AutoModelForCausalLM.from_pretrained(hf_model_name).cuda()
    model.eval()
    tok = AutoTokenizer.from_pretrained(hf_model_name)
    tok.pad_token = tok.eos_token
    return model, tok


def _benchmark_one_restart(
    *,
    model,
    tok,
    hparams: MEMITHyperParams,
    pool: List[Dict[str, Any]],
    ns: List[int],
    seed: int,
    initial_batch_size: int,
    outlier_upper_quantile: float,
) -> Dict[str, Any]:
    target_layer = int(hparams.layers[-1])
    max_n = max(ns)
    ns_sorted = sorted(set(int(x) for x in ns))
    if ns_sorted[0] <= 0:
        raise ValueError("All N must be > 0")
    if ns_sorted[-1] != max_n:
        raise ValueError("Internal error: max_n mismatch")

    sample = _sample_requests(pool, n=max_n, seed=seed)

    cur_batch_size = int(initial_batch_size)
    if cur_batch_size <= 0:
        raise ValueError("initial_batch_size must be > 0")

    norms: List[float] = []
    times_s: Dict[int, float] = {}
    means: Dict[int, Dict[str, Any]] = {}

    idx = 0
    next_idx_ptr = 0
    oom_retries = 0
    min_batch_size_used = cur_batch_size

    _sync()
    t0 = time.perf_counter()

    while idx < max_n:
        next_target = ns_sorted[next_idx_ptr]
        if next_target <= idx:
            next_idx_ptr += 1
            if next_idx_ptr >= len(ns_sorted):
                next_target = max_n
            else:
                next_target = ns_sorted[next_idx_ptr]

        bs = min(cur_batch_size, next_target - idx)
        batch = sample[idx : idx + bs]
        try:
            batch_norms = _compute_vstar_norms_batched(
                model=model,
                tok=tok,
                requests=batch,
                hparams=hparams,
                layer=target_layer,
            )
        except RuntimeError as e:
            if _is_cuda_oom(e) and bs > 1:
                oom_retries += 1
                new_bs = max(1, bs // 2)
                print(f"CUDA OOM at batch_size={bs}; retrying with batch_size={new_bs}")
                cur_batch_size = new_bs
                min_batch_size_used = min(min_batch_size_used, cur_batch_size)
                _recover_from_oom()
                continue
            raise

        norms.extend([float(x) for x in batch_norms])
        idx += len(batch_norms)

        if idx == next_target:
            _sync()
            elapsed = time.perf_counter() - t0
            times_s[int(next_target)] = float(elapsed)
            mean_trim, drop, threshold = _upper_trimmed_mean(
                norms, upper_quantile=float(outlier_upper_quantile)
            )
            means[int(next_target)] = {
                "raw_mean": float(np.mean(np.asarray(norms, dtype=np.float64))),
                "trimmed_mean": float(mean_trim),
                "drop_count": int(drop),
                "drop_threshold": float(threshold) if threshold is not None else None,
            }
            next_idx_ptr += 1

    if len(norms) != max_n:
        raise RuntimeError(f"Internal error: expected {max_n} norms, got {len(norms)}")

    meta = {
        "seed": int(seed),
        "target_layer": int(target_layer),
        "initial_batch_size": int(initial_batch_size),
        "final_batch_size": int(cur_batch_size),
        "min_batch_size_used": int(min_batch_size_used),
        "oom_retries": int(oom_retries),
    }
    return {"meta": meta, "times_s": times_s, "means": means}


def _summarize(results_by_restart: List[Dict[str, Any]], *, ns: List[int]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"by_n": {}}
    for n in sorted(set(int(x) for x in ns)):
        times: List[float] = []
        trimmed_means: List[float] = []
        raw_means: List[float] = []
        for r in results_by_restart:
            times_s = r.get("times_s", {})
            means = r.get("means", {})
            if not isinstance(times_s, dict) or not isinstance(means, dict):
                raise TypeError("Invalid benchmark record structure")
            t = times_s[str(n)] if str(n) in times_s else times_s[n]
            m = means[str(n)] if str(n) in means else means[n]
            if not isinstance(m, dict):
                raise TypeError("Invalid means structure")
            times.append(float(t))
            trimmed_means.append(float(m["trimmed_mean"]))
            raw_means.append(float(m["raw_mean"]))

        arr = np.asarray(times, dtype=np.float64)
        mean = float(arr.mean())
        var_pop = float(arr.var(ddof=0)) if arr.size > 0 else 0.0
        var_sample = float(arr.var(ddof=1)) if arr.size > 1 else 0.0

        trimmed_arr = np.asarray(trimmed_means, dtype=np.float64)
        raw_arr = np.asarray(raw_means, dtype=np.float64)
        summary["by_n"][str(n)] = {
            "time_mean_s": mean,
            "time_var_pop_s2": var_pop,
            "time_var_sample_s2": var_sample,
            "time_std_pop_s": float(np.sqrt(var_pop)) if var_pop >= 0 else None,
            "time_std_sample_s": float(np.sqrt(var_sample)) if var_sample >= 0 else None,
            "times_s": [float(x) for x in times],
            "trimmed_mean_vstar_norm_mean": float(trimmed_arr.mean()) if trimmed_arr.size > 0 else None,
            "trimmed_mean_vstar_norm_var_pop": float(trimmed_arr.var(ddof=0))
            if trimmed_arr.size > 0
            else None,
            "raw_mean_vstar_norm_mean": float(raw_arr.mean()) if raw_arr.size > 0 else None,
            "raw_mean_vstar_norm_var_pop": float(raw_arr.var(ddof=0)) if raw_arr.size > 0 else None,
        }
    return summary


def main() -> None:
    ns = list(NS)
    if not ns:
        raise ValueError("NS is empty")
    if any(int(x) <= 0 for x in ns):
        raise ValueError("All NS values must be > 0")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"vstar_norm_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"

    print(f"Writing results to: {out_dir}")

    config = {
        "models": [
            {
                "label": m["label"],
                "hf_model_name": m["hf_model_name"],
                "memit_hparams_path": str(m["memit_hparams_path"]),
            }
            for m in MODELS
        ],
        "ns": [int(x) for x in ns],
        "restarts": int(RESTARTS),
        "base_seed": int(BASE_SEED),
        "initial_batch_size": int(INITIAL_BATCH_SIZE),
        "outlier_upper_quantile": float(OUTLIER_UPPER_QUANTILE),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(config_path, config)

    pool = _build_request_pool()
    print(f"Built request pool with {len(pool)} samples")

    all_summary: Dict[str, Any] = {"out_dir": str(out_dir), "models": {}}

    for model_cfg in MODELS:
        label = str(model_cfg["label"])
        hf_model_name = str(model_cfg["hf_model_name"])
        memit_hparams_path = Path(model_cfg["memit_hparams_path"])
        if not memit_hparams_path.exists():
            raise FileNotFoundError(f"Missing MEMIT hparams at {memit_hparams_path}")

        hparams = MEMITHyperParams.from_json(memit_hparams_path)
        model, tok = _load_model_and_tok(hf_model_name)

        model_results: List[Dict[str, Any]] = []
        try:
            for restart_idx in range(int(RESTARTS)):
                seed = int(BASE_SEED) + int(restart_idx)
                print(f"\n[{label}] restart {restart_idx + 1}/{RESTARTS} (seed={seed})")
                rec = _benchmark_one_restart(
                    model=model,
                    tok=tok,
                    hparams=hparams,
                    pool=pool,
                    ns=ns,
                    seed=seed,
                    initial_batch_size=int(INITIAL_BATCH_SIZE),
                    outlier_upper_quantile=float(OUTLIER_UPPER_QUANTILE),
                )
                record = {
                    "model_label": label,
                    "hf_model_name": hf_model_name,
                    "memit_hparams_path": str(memit_hparams_path),
                    "restart_idx": int(restart_idx),
                    "meta": rec["meta"],
                    "times_s": rec["times_s"],
                    "means": rec["means"],
                }
                _append_jsonl(results_jsonl, record)
                model_results.append(record)
        finally:
            del model
            del tok
            _recover_from_oom()

        summary = _summarize(model_results, ns=ns)
        all_summary["models"][label] = {
            "hf_model_name": hf_model_name,
            "memit_hparams_path": str(memit_hparams_path),
            "restarts": int(RESTARTS),
            "summary": summary,
        }
        _atomic_write_json(summary_path, all_summary)

    print(f"Done. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
