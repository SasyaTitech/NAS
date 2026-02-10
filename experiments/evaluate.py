import os
import ast
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
import shutil
import random
from pathlib import Path
from itertools import islice
from time import time
from typing import List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    WikiBigEditDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from experiments.py.eval_utils_wikibigedit import compute_rewrite_quality_wikibigedit
from baselines.memit import MEMITHyperParams, apply_memit_to_model
from baselines.memit.compute_z import get_module_input_output_at_words, compute_z
from baselines.memit.memit_main import get_context_templates
from baselines.memit.memit_seq_main import apply_memit_seq_to_model
from baselines.memit.memit_rect_main import apply_memit_rect_to_model
from baselines.AlphaEdit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from baselines.AlphaEdit.AlphaEdit_main import get_cov as get_cov_alphaedit
from baselines.encore import ENCOREHyperParams, apply_encore_to_model
from NAS import NASHyperParams
from NAS.NAS_main import apply_NAS_to_model, get_cov as get_cov_nas
from baselines.lyaplock import LyapLockHyperParams, apply_lyaplock_to_model
from baselines.rome import ROMEHyperParams, apply_rome_to_model
from baselines.memoir import MEMOIRHyperParams, apply_memoir_to_model
from util import nethook
from util.globals import *
from baselines.nse import NSEHyperParams, apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "NAS": (NASHyperParams, apply_NAS_to_model),
    "ENCORE": (ENCOREHyperParams, apply_encore_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "NSE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "LyapLock": (LyapLockHyperParams, apply_lyaplock_to_model),
    "MEMOIR": (MEMOIRHyperParams, apply_memoir_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
    "wikibigedit": (WikiBigEditDataset, compute_rewrite_quality_wikibigedit),
}


class _OnlineMoments:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, value) -> None:
        if value is None:
            return
        x = float(value)
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    def mean_std(self) -> Optional[Tuple[float, float]]:
        if self.n == 0:
            return None
        mean = self.sum / self.n
        var = max(0.0, (self.sumsq / self.n) - (mean * mean))
        return mean, float(np.sqrt(var))


def _harmonic_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def _blockwise_nested_sample_indices(
    *,
    after_edits: int,
    block_size: int,
    sample_ratio: float,
    seed: int,
) -> List[int]:
    """
    Deterministic, nested sampling scheme:
    - Split [0, after_edits) into fixed-size blocks of length `block_size`.
    - In each block, sample `floor(block_len * sample_ratio)` indices using a per-block RNG seed.
    - Union across blocks and sort, so checkpoints are nested across increasing after_edits.
    """
    if after_edits < 0:
        raise ValueError("after_edits must be >= 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1]")

    indices: List[int] = []
    n_blocks = (after_edits + block_size - 1) // block_size
    for block_idx in range(n_blocks):
        start = block_idx * block_size
        end = min(start + block_size, after_edits)
        block_len = end - start
        k = int(block_len * sample_ratio)
        if k <= 0 and block_len > 0:
            k = 1
        if k <= 0:
            continue
        h = hashlib.sha256(f"{seed}:{block_idx}".encode("utf-8")).digest()
        block_seed = int.from_bytes(h[:8], "big", signed=False)
        rng = random.Random(block_seed)
        offsets = rng.sample(range(block_len), k=k)
        indices.extend([start + off for off in offsets])

    indices = sorted(set(indices))
    if not indices and after_edits > 0:
        indices = [0]
    return indices


def _summarize_checkpoint_metrics(after_edits: int, records_iter) -> dict:
    """
    Summarize the streaming checkpoint-eval records in the same spirit as
    experiments/summarize.py, but without re-reading results.jsonl.
    """
    stats = defaultdict(_OnlineMoments)

    for data in records_iter:
        post = data.get("post") or {}

        for src_key, base in [
            ("rewrite_prompts_probs", "rewrite"),
            ("paraphrase_prompts_probs", "paraphrase"),
        ]:
            vals = post.get(src_key)
            if vals:
                stats[f"post_{base}_success"].update(
                    np.mean([x["target_true"] > x["target_new"] for x in vals])
                )
                stats[f"post_{base}_diff"].update(
                    np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in vals])
                )

        vals = post.get("neighborhood_prompts_probs")
        if vals:
            stats["post_neighborhood_success"].update(
                np.mean([x["target_true"] < x["target_new"] for x in vals])
            )
            stats["post_neighborhood_diff"].update(
                np.mean([np.exp(-x["target_true"]) - np.exp(-x["target_new"]) for x in vals])
            )

        for src_key, dst_key in [
            ("rewrite_prompts_correct", "post_rewrite_acc"),
            ("paraphrase_prompts_correct", "post_paraphrase_acc"),
            ("neighborhood_prompts_correct", "post_neighborhood_acc"),
        ]:
            vals = post.get(src_key)
            if vals is not None:
                stats[dst_key].update(np.mean(vals))

        for key in ["ngram_entropy", "reference_score", "essence_score"]:
            if key in post:
                stats[f"post_{key}"].update(post[key])

        for key in ["ES", "GS", "LS"]:
            val = post.get(key)
            if isinstance(val, (int, float, np.generic)):
                stats[f"post_{key}"].update(val)

    summary = {}
    for k, v in stats.items():
        ms = v.mean_std()
        if ms is None:
            continue
        mean, std = ms
        if all(exclude not in k for exclude in ["essence_score", "time"]):
            mean *= 100.0
            std *= 100.0
        summary[k] = (round(mean, 2), round(std, 2))

    score_keys = ["post_rewrite_success", "post_paraphrase_success", "post_neighborhood_success"]
    if all(k in summary for k in score_keys):
        score = _harmonic_mean([summary[k][0] for k in score_keys])
        summary["post_score"] = (round(score, 2), None)

    return {
        "after_edits": after_edits,
        "num_cases": after_edits,
        "metrics": summary,
    }


def _filter_unsupported_kwargs(func, kwargs):
    """
    Keep evaluate.py backwards-compatible when passing optional args into editing
    functions that haven't been updated yet.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        return dict(kwargs)
    supported = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in supported}


def _parse_override_value(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _projection_cache_key(*, hf_model_name: str, hparams, namespace: str) -> str:
    payload = {
        "version": 2,
        "namespace": namespace,
        "hf_model_name": hf_model_name,
        "layers": list(getattr(hparams, "layers", [])),
        "rewrite_module_tmp": getattr(hparams, "rewrite_module_tmp", None),
        "mom2_dataset": getattr(hparams, "mom2_dataset", None),
        "mom2_n_samples": getattr(hparams, "mom2_n_samples", None),
        "mom2_dtype": getattr(hparams, "mom2_dtype", None),
        "nullspace_threshold": getattr(hparams, "nullspace_threshold", None),
    }

    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    digest = hashlib.sha1(blob).hexdigest()[:12]

    safe_model = "".join(
        c if (c.isalnum() or c in "._-") else "_"
        for c in hf_model_name.replace("/", "_").replace(":", "_")
    )
    safe_model = safe_model[:80] if safe_model else "model"
    return f"{safe_model}_{digest}"


def _legacy_projection_cache_key(*, hf_model_name: str, hparams, get_cov_fn) -> str:
    try:
        cov_impl = f"{get_cov_fn.__module__}.{get_cov_fn.__qualname__}"
    except Exception:
        cov_impl = repr(get_cov_fn)

    payload = {
        "version": 1,
        "hf_model_name": hf_model_name,
        "layers": list(getattr(hparams, "layers", [])),
        "rewrite_module_tmp": getattr(hparams, "rewrite_module_tmp", None),
        "mom2_dataset": getattr(hparams, "mom2_dataset", None),
        "mom2_n_samples": getattr(hparams, "mom2_n_samples", None),
        "mom2_dtype": getattr(hparams, "mom2_dtype", None),
        "nullspace_threshold": getattr(hparams, "nullspace_threshold", None),
        "cov_impl": cov_impl,
    }

    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    digest = hashlib.sha1(blob).hexdigest()[:12]

    safe_model = "".join(
        c if (c.isalnum() or c in "._-") else "_"
        for c in hf_model_name.replace("/", "_").replace(":", "_")
    )
    safe_model = safe_model[:80] if safe_model else "model"
    return f"{safe_model}_{digest}"


def _global_projection_cache_path(*, hf_model_name: str, hparams, namespace: str) -> Path:
    return (
        Path(STATS_DIR)
        / "null_space_project"
        / f"{_projection_cache_key(hf_model_name=hf_model_name, hparams=hparams, namespace=namespace)}.pt"
    )


def _legacy_global_projection_cache_paths(*, hf_model_name: str, hparams) -> list[Path]:
    cache_dir = Path(STATS_DIR) / "null_space_project"
    return [
        cache_dir
        / f"{_legacy_projection_cache_key(hf_model_name=hf_model_name, hparams=hparams, get_cov_fn=cov_fn)}.pt"
        for cov_fn in (get_cov_alphaedit, get_cov_nas)
    ]


def _projection_shape_ok(P, *, n_layers: int, k_dim: int) -> bool:
    if not torch.is_tensor(P):
        return False
    if P.ndim != 3:
        return False
    return P.shape == (n_layers, k_dim, k_dim)


def _normalize_case_id(case_id):
    if isinstance(case_id, np.generic):
        return int(case_id)
    if case_id is None:
        return None
    try:
        return int(case_id)
    except Exception:
        return case_id


_RESUME_CONTEXT_TEMPLATE_MODULES = {
    "AlphaEdit": "baselines.AlphaEdit.AlphaEdit_main",
    "NAS": "NAS.NAS_main",
    "ENCORE": "baselines.encore.encore_main",
    "MEMIT": "baselines.memit.memit_main",
    "MEMIT_prune": "baselines.memit.memit_main",
    "MEMIT_rect": "baselines.memit.memit_rect_main",
    "MEMIT_seq": "baselines.memit.memit_seq_main",
    "NSE": "baselines.nse.nse_main",
    "ROME": "baselines.rome.rome_main",
    "LyapLock": "baselines.lyaplock.lyaplock_main",
}

_RESUMABLE_ALGOS = {
    "AlphaEdit",
    "NAS",
    "ENCORE",
    "ROME",
    "MEMIT",
    "MEMIT_prune",
    "MEMIT_rect",
    "FT",
    "MEMIT_seq",
    "NSE",
    "LyapLock",
}


def _get_context_templates_cache(alg_name: str):
    module_path = _RESUME_CONTEXT_TEMPLATE_MODULES.get(alg_name)
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return None
    return getattr(module, "CONTEXT_TEMPLATES_CACHE", None)


def _collect_resume_weights(*, alg_name: str, hparams, model) -> dict:
    if alg_name == "FT":
        module_prefixes = [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers]
        weights_to_save = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if any(prefix in name for prefix in module_prefixes):
                    weights_to_save[name] = param.detach().cpu()
        return weights_to_save

    weights_to_save = {}
    with torch.no_grad():
        for layer in hparams.layers:
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            weights_to_save[weight_name] = (
                nethook.get_parameter(model, weight_name).detach().cpu()
            )
    return weights_to_save


def _save_resume_state(
    *,
    run_dir: Path,
    alg_name: str,
    hf_model_name: str,
    ds_name: str,
    dataset_size_limit: Optional[int],
    num_edits: int,
    use_cache: bool,
    skip_generation_tests: bool,
    generation_test_interval: int,
    edit_log: bool,
    checkpoint_eval_interval: int,
    save_edited_weights_interval: int,
    wikibigedit_checkpoint_eval_sample_ratio: Optional[float] = None,
    wikibigedit_checkpoint_eval_sample_seed: int = 0,
    downstream_eval_steps: int,
    after_edits: int,
    cnt: int,
    checkpoint_eval_count: int,
    last_checkpoint_after: int,
    edited_records: List[dict],
    hparams,
    model,
    cache_c,
    resume_weights: Optional[dict] = None,
    lyaplock_kwargs: Optional[dict] = None,
) -> None:
    """
    Save the minimal state needed to continue editing in a later job.
    This is intentionally decoupled from save_edited_weights_interval (debug snapshots).
    """
    resume_dir = run_dir / "resume"
    projection_path = None
    if alg_name in {"AlphaEdit", "NAS"}:
        state_dir_name = "alphaedit_state" if alg_name == "AlphaEdit" else "nas_state"
        projection_path = (run_dir / state_dir_name) / "null_space_project.pt"

    weights_to_save = (
        resume_weights
        if resume_weights is not None
        else _collect_resume_weights(alg_name=alg_name, hparams=hparams, model=model)
    )
    weights_blob = {"after_edits": after_edits, "weights": weights_to_save}
    if hasattr(hparams, "layers"):
        weights_blob["layers"] = list(hparams.layers)
    if hasattr(hparams, "rewrite_module_tmp"):
        weights_blob["rewrite_module_tmp"] = hparams.rewrite_module_tmp
    _atomic_torch_save(
        resume_dir / "rewrite_module_weights.pt",
        weights_blob,
    )

    if cache_c is not None and alg_name in {"AlphaEdit", "NAS", "MEMIT_seq", "NSE"}:
        _atomic_torch_save(resume_dir / "cache_c.pt", cache_c)

    context_templates = _get_context_templates_cache(alg_name)
    if context_templates is not None:
        _atomic_write_json(resume_dir / "context_templates.json", context_templates)

    lyaplock_state_path: Optional[Path] = None
    if alg_name == "LyapLock" and lyaplock_kwargs is not None:
        try:
            repo_root = Path(__file__).resolve().parents[2]
            run_ref = str(run_dir.resolve().relative_to(repo_root.resolve()))
        except Exception:
            run_ref = str(run_dir.resolve())

        payload = {
            "version": 1,
            "algo": alg_name,
            "hf_model_name": hf_model_name,
            "ds_name": ds_name,
            "run_ref": run_ref,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
        digest = hashlib.sha1(blob).hexdigest()[:12]
        safe_model = "".join(
            c if (c.isalnum() or c in "._-") else "_"
            for c in hf_model_name.replace("/", "_").replace(":", "_")
        )
        safe_model = safe_model[:80] if safe_model else "model"
        safe_ds = "".join(c if (c.isalnum() or c in "._-") else "_" for c in ds_name)
        safe_ds = safe_ds[:40] if safe_ds else "ds"
        lyaplock_state_path = (
            Path(STATS_DIR)
            / "resume_states"
            / "lyaplock"
            / f"{safe_model}_{safe_ds}_{digest}.pt"
        )
        _atomic_torch_save(lyaplock_state_path, lyaplock_kwargs)
        print(f"Saved LyapLock state to {lyaplock_state_path}")

    if alg_name == "ENCORE":
        try:
            module = importlib.import_module("baselines.encore.encore_main")
            seq_cache = getattr(module, "SEQ_CACHE", None)
        except Exception:
            seq_cache = None
        if seq_cache is not None:
            _atomic_torch_save(resume_dir / "encore_seq_cache.pt", seq_cache)

    edited_case_ids = [_normalize_case_id(r.get("case_id")) for r in edited_records]
    state = {
        "version": 1,
        "algo": alg_name,
        "cwd": os.getcwd(),
        "hf_model_name": hf_model_name,
        "ds_name": ds_name,
        "dataset_size_limit": dataset_size_limit,
        "num_edits": num_edits,
        "use_cache": use_cache,
        "skip_generation_tests": skip_generation_tests,
        "generation_test_interval": generation_test_interval,
        "edit_log": edit_log,
        "checkpoint_eval_interval": checkpoint_eval_interval,
        "save_edited_weights_interval": save_edited_weights_interval,
        "wikibigedit_checkpoint_eval_sample_ratio": wikibigedit_checkpoint_eval_sample_ratio,
        "wikibigedit_checkpoint_eval_sample_seed": wikibigedit_checkpoint_eval_sample_seed,
        "downstream_eval_steps": downstream_eval_steps,
        "progress": {
            "after_edits": after_edits,
            "cnt": cnt,
            "checkpoint_eval_count": checkpoint_eval_count,
            "last_checkpoint_after": last_checkpoint_after,
            "edited_case_ids": edited_case_ids,
        },
        "files": {
            "rewrite_module_weights": str(
                (resume_dir / "rewrite_module_weights.pt").relative_to(run_dir)
            ),
            "hparams": str((run_dir / "params.json").relative_to(run_dir)),
        },
    }
    if (resume_dir / "cache_c.pt").exists():
        state["files"]["cache_c"] = str((resume_dir / "cache_c.pt").relative_to(run_dir))
    if (resume_dir / "context_templates.json").exists():
        state["files"]["context_templates"] = str(
            (resume_dir / "context_templates.json").relative_to(run_dir)
        )
    if lyaplock_state_path is not None and lyaplock_state_path.exists():
        state["files"]["lyaplock_state"] = os.path.relpath(
            lyaplock_state_path, start=run_dir
        )
    if (resume_dir / "encore_seq_cache.pt").exists():
        state["files"]["encore_seq_cache"] = str(
            (resume_dir / "encore_seq_cache.pt").relative_to(run_dir)
        )
    if projection_path is not None and projection_path.exists():
        state["files"]["projection_matrix"] = str(projection_path.relative_to(run_dir))
    _atomic_write_json(resume_dir / "state.json", state)
    print(f"Saved resume state to {resume_dir}")


def _apply_hparam_overrides(hparams, overrides: List[str]):
    if not overrides:
        return {}
    before = asdict(hparams)
    override_keys: List[str] = []
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Invalid --hparam {spec!r}; expected key=value")
        key, raw = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --hparam {spec!r}; empty key")
        if key not in before:
            raise ValueError(
                f"Unknown hyperparameter {key!r} for {type(hparams).__name__}"
            )
        setattr(hparams, key, _parse_override_value(raw))
        override_keys.append(key)

    after = asdict(hparams)
    return {k: {"before": before.get(k), "after": after.get(k)} for k in override_keys}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    edit_log: bool = False,
    checkpoint_eval_interval: int = 0,
    save_edited_weights_interval: int = 0,
    wikibigedit_checkpoint_eval_sample_ratio: Optional[float] = None,
    wikibigedit_checkpoint_eval_sample_seed: int = 0,
    hparam_overrides: Optional[List[str]] = None,
    downstream_eval_steps: int = 0,
    resume_state: Optional[dict] = None,
    run_dir_override: Optional[Union[str, Path]] = None,
    memit_prune_base_weights: Optional[dict] = None,
    lyaplock_alpha: float = 60.0,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # MEMOIR: always single-edit + skip generation tests (KV-cache decoding is not supported)
    if alg_name == "MEMOIR":
        if num_edits != 1:
            print(f"[MEMOIR] Forcing --num_edits=1 (got {num_edits})")
            num_edits = 1
        if not skip_generation_tests:
            print("[MEMOIR] Forcing --skip_generation_tests")
        skip_generation_tests = True
        generation_test_interval = -1

    # WikiBigEdit: generation tests are not defined in this setting.
    if ds_name == "wikibigedit":
        if not skip_generation_tests:
            print("[WikiBigEdit] Forcing --skip_generation_tests")
        skip_generation_tests = True
        generation_test_interval = -1
        if wikibigedit_checkpoint_eval_sample_ratio is None:
            wikibigedit_checkpoint_eval_sample_ratio = 0.1
            print(
                "[WikiBigEdit] Defaulting --wikibigedit_checkpoint_eval_sample_ratio=0.1 "
                "(override via CLI to change/disable sampling)."
            )
    else:
        if wikibigedit_checkpoint_eval_sample_ratio is not None:
            raise ValueError(
                "--wikibigedit_checkpoint_eval_sample_ratio is only valid for --ds_name wikibigedit"
            )
        if wikibigedit_checkpoint_eval_sample_seed != 0:
            raise ValueError(
                "--wikibigedit_checkpoint_eval_sample_seed is only valid for --ds_name wikibigedit"
            )

    if wikibigedit_checkpoint_eval_sample_ratio is not None and not (
        0.0 < float(wikibigedit_checkpoint_eval_sample_ratio) <= 1.0
    ):
        raise ValueError("--wikibigedit_checkpoint_eval_sample_ratio must be in (0, 1]")

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if run_dir_override is not None:
        run_dir = Path(run_dir_override)
        run_dir.mkdir(parents=True, exist_ok=True)
        continuing = (run_dir / "params.json").exists()
    else:
        if (
            continue_from_run is None
            or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
        ):
            continue_from_run = None
        continuing = continue_from_run is not None
        if continue_from_run is None:
            alg_dir = RESULTS_DIR / dir_name
            if alg_dir.exists():
                id_list = [
                    int(str(x).split("_")[-1])
                    for x in alg_dir.iterdir()
                    if str(x).split("_")[-1].isnumeric()
                ]
                run_id = 0 if not id_list else max(id_list) + 1
            else:
                run_id = 0
            run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
            run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    edit_log_kwargs = {}
    if edit_log:
        edit_log_dir = run_dir / "edit_logs"
        edit_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Edit logs directory: {edit_log_dir}")
        edit_log_kwargs = _filter_unsupported_kwargs(
            apply_algo, dict(edit_log_dir=str(edit_log_dir))
        )
    # Get run hyperparameters
    if "MEMIT" in alg_name:
        params_path = (
            run_dir / "params.json" if continuing else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json" if continuing else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    hparam_overrides = hparam_overrides or []
    if hparam_overrides and continuing:
        raise ValueError("Cannot use --hparam when continuing an existing run")
    if hparam_overrides:
        changes = _apply_hparam_overrides(hparams, hparam_overrides)
        shutil.copyfile(params_path, run_dir / "params_base.json")
        with open(run_dir / "params.json", "w") as f:
            json.dump(_to_jsonable(asdict(hparams)), f, indent=2)
        with open(run_dir / "hparam_overrides.json", "w") as f:
            json.dump(
                {
                    "base_params": str(params_path),
                    "overrides": hparam_overrides,
                    "changes": _to_jsonable(changes),
                },
                f,
                indent=2,
            )
        print(f"Applied hyperparameter overrides: {changes}")
    elif not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    hf_model_name = None
    if type(model_name) is str:
        print("Instantiating model")
        hf_model_name = model_name
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
        hf_model_name = model_name

    memit_prune_base_weights_local: Optional[dict] = None
    if alg_name == "MEMIT_prune":
        if resume_state is not None:
            if memit_prune_base_weights is None:
                raise ValueError(
                    "MEMIT_prune resume requires memit_prune_base_weights (base rewrite-module weights). "
                    "Use experiments/resume_run.py to resume MEMIT_prune runs."
                )
            memit_prune_base_weights_local = memit_prune_base_weights
        else:
            memit_prune_base_weights_local = {}
            with torch.no_grad():
                for layer in hparams.layers:
                    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                    memit_prune_base_weights_local[weight_name] = (
                        nethook.get_parameter(model, weight_name).detach().cpu()
                    )

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None
    gen_test_vars = [snips, vec]

    checkpoint_root = run_dir / "checkpoint_evals"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    edited_records: List[dict] = []
    resume_progress = (
        resume_state.get("progress", resume_state) if resume_state is not None else {}
    )
    last_checkpoint_after = int(resume_progress.get("last_checkpoint_after", 0))
    checkpoint_eval_count = int(resume_progress.get("checkpoint_eval_count", 0))
    resume_after_edits = int(resume_progress.get("after_edits", 0))
    resume_cnt = int(resume_progress.get("cnt", 0))
    resume_cache_c = resume_progress.get("cache_c")
    resume_case_ids = resume_progress.get("edited_case_ids")
    resume_lyaplock_kwargs = resume_progress.get("lyaplock_kwargs")

    if save_edited_weights_interval < 0:
        raise ValueError("--save_edited_weights_interval must be >= 0")
    if save_edited_weights_interval > 0 and checkpoint_eval_interval <= 0:
        print(
            "Warning: --save_edited_weights_interval is set but --checkpoint_eval_interval is 0; "
            "skipping weight saves."
        )
        save_edited_weights_interval = 0

    def run_checkpoint_eval(*, after_edits: int, model_to_eval):
        nonlocal last_checkpoint_after, checkpoint_eval_count
        checkpoint_eval_count += 1
        ckpt_dir = checkpoint_root / f"after_{after_edits}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_path = ckpt_dir / "results.jsonl"

        sample_indices: Optional[List[int]] = None
        if ds_name == "wikibigedit" and wikibigedit_checkpoint_eval_sample_ratio is not None:
            block_size = (
                checkpoint_eval_interval * num_edits
                if checkpoint_eval_interval > 0
                else after_edits
            )
            sample_indices = _blockwise_nested_sample_indices(
                after_edits=after_edits,
                block_size=block_size,
                sample_ratio=float(wikibigedit_checkpoint_eval_sample_ratio),
                seed=int(wikibigedit_checkpoint_eval_sample_seed),
            )
            _atomic_write_json(ckpt_dir / "sample_indices.json", sample_indices)

        if sample_indices is not None:
            print(
                f"Running checkpoint eval after {after_edits} edited records "
                f"(sampled {len(sample_indices)} cases) -> {out_path}"
            )
        else:
            print(f"Running checkpoint eval after {after_edits} edited records -> {out_path}")
        start_eval = time()
        with open(out_path, "w") as f:
            def _iter_records():
                eval_order = (
                    sample_indices
                    if sample_indices is not None
                    else list(range(len(edited_records)))
                )
                for eval_idx in eval_order:
                    record = edited_records[eval_idx]
                    do_gen = (
                        snips is not None
                        and vec is not None
                        and generation_test_interval is not None
                        and generation_test_interval > 0
                        and (eval_idx % generation_test_interval == 0)
                    )
                    post_metrics = ds_eval_method(
                        model_to_eval,
                        tok,
                        record,
                        *(gen_test_vars if do_gen else [None, None]),
                    )
                    metrics = {
                        "case_id": record.get("case_id"),
                        "edit_order_idx": eval_idx,
                        "requested_rewrite": record.get("requested_rewrite"),
                        "post": post_metrics,
                    }
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                    yield metrics

            ckpt_summary = _summarize_checkpoint_metrics(after_edits, _iter_records())
        print("Checkpoint evaluation took", time() - start_eval)
        _atomic_write_json(
            ckpt_dir / "summary.json",
            {
                "checkpoint_eval_idx": checkpoint_eval_count,
                "results_path": str(out_path.relative_to(run_dir)),
                **(
                    {
                        "eval_sample": {
                            "enabled": True,
                            "scheme": "blockwise_nested",
                            "ratio": float(wikibigedit_checkpoint_eval_sample_ratio),
                            "seed": int(wikibigedit_checkpoint_eval_sample_seed),
                            "count": len(sample_indices),
                            "indices_path": str((ckpt_dir / "sample_indices.json").relative_to(run_dir)),
                        }
                    }
                    if sample_indices is not None
                    else {}
                ),
                **ckpt_summary,
            },
        )
        summary_metrics = ckpt_summary.get("metrics", {})
        summary_keys = [
            "post_score",
            "post_rewrite_success",
            "post_paraphrase_success",
            "post_neighborhood_success",
            "post_rewrite_acc",
            "post_paraphrase_acc",
            "post_neighborhood_acc",
        ]
        parts = []
        for k in summary_keys:
            v = summary_metrics.get(k)
            if isinstance(v, (list, tuple)) and v:
                parts.append(f"{k}={v[0]}")
        if parts:
            print(f"Checkpoint summary(after_{after_edits}): " + ", ".join(parts))
            print(f"Wrote checkpoint summary -> {ckpt_dir / 'summary.json'}")

        # Persist resume state at each checkpoint so preemption/time-limits do not
        # rewind progress back to the last end-of-run save.
        if alg_name in _RESUMABLE_ALGOS:
            _save_resume_state(
                run_dir=run_dir,
                alg_name=alg_name,
                hf_model_name=hf_model_name,
                ds_name=ds_name,
                dataset_size_limit=dataset_size_limit,
                num_edits=num_edits,
                use_cache=use_cache,
                skip_generation_tests=skip_generation_tests,
                generation_test_interval=generation_test_interval,
                edit_log=edit_log,
                checkpoint_eval_interval=checkpoint_eval_interval,
                save_edited_weights_interval=save_edited_weights_interval,
                wikibigedit_checkpoint_eval_sample_ratio=wikibigedit_checkpoint_eval_sample_ratio,
                wikibigedit_checkpoint_eval_sample_seed=wikibigedit_checkpoint_eval_sample_seed,
                downstream_eval_steps=downstream_eval_steps,
                after_edits=after_edits,
                cnt=cnt,
                checkpoint_eval_count=checkpoint_eval_count,
                last_checkpoint_after=after_edits,
                edited_records=edited_records,
                hparams=hparams,
                model=model_to_eval,
                cache_c=cache_c,
                lyaplock_kwargs=lyaplock_kwargs if alg_name == "LyapLock" else None,
            )

        if (
            save_edited_weights_interval > 0
            and checkpoint_eval_interval > 0
            and (checkpoint_eval_count % save_edited_weights_interval == 0)
        ):
            if not (hasattr(hparams, "layers") and hasattr(hparams, "rewrite_module_tmp")):
                print(
                    "Skipping --save_edited_weights_interval: this algorithm does not expose "
                    "`layers`/`rewrite_module_tmp` in its hparams."
                )
                last_checkpoint_after = after_edits
                return
            weights_dir = run_dir / "edited_weights" / f"after_{after_edits}"
            weights_dir.mkdir(parents=True, exist_ok=True)
            weights_to_save = {}
            with torch.no_grad():
                for layer in hparams.layers:
                    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                    weights_to_save[weight_name] = (
                        nethook.get_parameter(model_to_eval, weight_name).detach().cpu()
                    )
            torch.save(
                {
                    "checkpoint_eval_idx": checkpoint_eval_count,
                    "after_edits": after_edits,
                    "layers": list(hparams.layers),
                    "rewrite_module_tmp": hparams.rewrite_module_tmp,
                    "weights": weights_to_save,
                },
                weights_dir / "rewrite_module_weights.pt",
            )
            print(
                f"Saved edited weights at {weights_dir / 'rewrite_module_weights.pt'} "
                f"(checkpoint #{checkpoint_eval_count})"
            )
        last_checkpoint_after = after_edits

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    if resume_state is not None:
        if resume_after_edits < 0 or resume_after_edits > len(ds):
            raise ValueError(
                f"Invalid resume after_edits={resume_after_edits}; dataset has {len(ds)} records"
            )
        edited_records = [ds[i] for i in range(resume_after_edits)]
        if resume_case_ids is not None:
            loaded_ids = [_normalize_case_id(r.get("case_id")) for r in edited_records]
            if loaded_ids != resume_case_ids:
                raise ValueError(
                    "Resume state mismatch: edited case_ids do not match current dataset order. "
                    "Make sure you resume with the same dataset + tokenizer + preprocessing."
                )
        ds = ds[resume_after_edits:]
    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    cache_c = None
    P = None
    if any(alg in alg_name for alg in ["AlphaEdit", "NAS", "MEMIT_seq", "MEMIT_prune", "NSE"]):
        # Iterate through dataset
        rewrite_module_name = hparams.rewrite_module_tmp.format(hparams.layers[-1])
        rewrite_module = nethook.get_module(model, rewrite_module_name)
        W_out = nethook.get_parameter(model, f"{rewrite_module_name}.weight")

        if hasattr(rewrite_module, "in_features"):
            k_dim = int(rewrite_module.in_features)
        elif rewrite_module.__class__.__name__ == "Conv1D":
            if W_out.ndim != 2:
                raise ValueError(
                    f"Expected 2D weight for {rewrite_module_name}, got shape {tuple(W_out.shape)}"
                )
            k_dim = int(W_out.shape[0])
        else:
            if W_out.ndim != 2:
                raise ValueError(
                    f"Expected 2D weight for {rewrite_module_name}, got shape {tuple(W_out.shape)}"
                )
            # Fall back to PyTorch Linear layout: (out_features, in_features).
            k_dim = int(W_out.shape[1])

        if resume_cache_c is not None:
            cache_c = resume_cache_c
        else:
            cache_c = torch.zeros((len(hparams.layers), k_dim, k_dim), device="cpu")
        del W_out
    if alg_name in {"AlphaEdit", "NAS"}:
        state_dir_name = "alphaedit_state" if alg_name == "AlphaEdit" else "nas_state"
        algo_state_dir = run_dir / state_dir_name
        algo_state_dir.mkdir(parents=True, exist_ok=True)
        projection_path = algo_state_dir / "null_space_project.pt"
        get_cov_fn = get_cov_alphaedit if alg_name == "AlphaEdit" else get_cov_nas
        projection_namespace = "alphaedit_family"
        global_projection_path = _global_projection_cache_path(
            hf_model_name=str(hf_model_name),
            hparams=hparams,
            namespace=projection_namespace,
        )
        legacy_projection_paths = _legacy_global_projection_cache_paths(
            hf_model_name=str(hf_model_name),
            hparams=hparams,
        )
        # Prefer global cache (DATA_DIR) to avoid duplicating multi-GB matrices inside results/.
        # Keep backward-compatibility: if a per-run file exists (older runs), it is still used.
        loaded_from: Optional[Path] = None
        if projection_path.exists():
            print(f"Loading cached projection matrix from {projection_path}")
            P = torch.load(projection_path, map_location="cpu")
            loaded_from = projection_path
            if not _projection_shape_ok(P, n_layers=len(hparams.layers), k_dim=k_dim):
                print(
                    f"Warning: cached projection shape mismatch at {projection_path}; "
                    "falling back to global cache/recompute."
                )
                P = None
                loaded_from = None

        if P is None:
            cache_candidates = [global_projection_path, *legacy_projection_paths]
            hit_path = next((p for p in cache_candidates if p.exists()), None)
            if hit_path is not None:
                print(f"Loading global cached projection matrix from {hit_path}")
                P = torch.load(hit_path, map_location="cpu")
                loaded_from = hit_path
                if _projection_shape_ok(P, n_layers=len(hparams.layers), k_dim=k_dim):
                    # Migrate legacy cache keys -> current key, but do NOT copy into results/.
                    if hit_path != global_projection_path:
                        global_projection_path.parent.mkdir(parents=True, exist_ok=True)
                        _atomic_torch_save(global_projection_path, P)
                        print(
                            f"Migrated legacy projection cache {hit_path} -> {global_projection_path}"
                        )
                else:
                    print(
                        f"Warning: cached projection shape mismatch at {hit_path}; recomputing."
                    )
                    P = None
                    loaded_from = None

        if P is None:
            P = torch.zeros((len(hparams.layers), k_dim, k_dim), device="cpu")
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(
                    model, tok, layer, hparams, get_cov_fn=get_cov_fn
                )
            global_projection_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(global_projection_path, P)
            loaded_from = global_projection_path
            print(f"Saved global projection matrix to {global_projection_path}")

        if loaded_from is not None and loaded_from == global_projection_path:
            print(f"Using global projection matrix at {global_projection_path}")

    lyaplock_kwargs = None
    if alg_name == "LyapLock":
        if resume_state is not None and resume_lyaplock_kwargs is not None:
            lyaplock_kwargs = resume_lyaplock_kwargs
        else:
            lyaplock_kwargs = {
                "V": {l: 1 for l in hparams.layers},
                "Z": {l: "1sqrtD" for l in hparams.layers},
                "a": {l: "1_sqrtD" for l in hparams.layers},
                "b": {l: 0 for l in hparams.layers},
                "zmax": {l: "1sqrtD" for l in hparams.layers},
                "alpha": {l: lyaplock_alpha for l in hparams.layers},
                "method_name": alg_name,
                "D_base": {l: None for l in hparams.layers},
                "Pre_Cache": {l: [] for l in hparams.layers},
            }
    # hs = get_module_input_output_at_words(
    #         model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "pre_edit_hs.pt")
    # del hs
    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    cnt = resume_cnt if resume_state is not None else 0
    edited_model = model
    for record_chunks in chunks(ds, num_edits):
        print(f"=================================================================={cnt+1}_edit==================================================================")
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        last_case_id = case_ids[-1] if case_ids else None
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","AlphaEdit", "NAS", "MEMIT_seq", "MEMIT_prune", "NSE", "LyapLock", "ENCORE"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "NAS", "MEMIT_seq", "NSE"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit", "NAS"]) else dict()
        if cnt == 0 and downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
            glue_results = {'edit_num': -1}

            out_file = glue_save_location + "base.json"
            
            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit", "NAS", "MEMIT_seq", "NSE"]):
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                **etc_args,
                **seq_args,
                **nc_args,
                **edit_log_kwargs,
            )
        elif alg_name == "LyapLock":
            if lyaplock_kwargs is None:
                raise RuntimeError("LyapLock kwargs not initialized")
            lyaplock_kwargs["cnt"] = cnt
            edited_model, _, lyaplock_kwargs = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **etc_args,
                **lyaplock_kwargs,
                **edit_log_kwargs,
            )
        elif alg_name == "MEMIT_prune":
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **etc_args,
                **edit_log_kwargs,
            )
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **etc_args,
                **edit_log_kwargs,
            )
        exec_time = time() - start
        edited_records.extend(record_chunks)
        cnt+=1
        print("Execution took", exec_time)
        # Evaluate new model
    
        if downstream_eval_steps > 0 and cnt % downstream_eval_steps == 0:
            glue_results = {
                        'edit_num': cnt*num_edits,
                        'case_id': case_ids
                        }

            out_file = glue_save_location + "case_{}.json".format(last_case_id)#stores the last case ID of the batch

            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    
            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        if checkpoint_eval_interval > 0 and cnt % checkpoint_eval_interval == 0:
            run_checkpoint_eval(after_edits=len(edited_records), model_to_eval=edited_model)
    # hs = get_module_input_output_at_words(
    #         edited_model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "post_edit_hs_memit.pt")
    if alg_name == "MEMIT_prune":
        if alg_name in _RESUMABLE_ALGOS:
            _save_resume_state(
                run_dir=run_dir,
                alg_name=alg_name,
                hf_model_name=hf_model_name,
                ds_name=ds_name,
                dataset_size_limit=dataset_size_limit,
                num_edits=num_edits,
                use_cache=use_cache,
                skip_generation_tests=skip_generation_tests,
                generation_test_interval=generation_test_interval,
                edit_log=edit_log,
                checkpoint_eval_interval=checkpoint_eval_interval,
                save_edited_weights_interval=save_edited_weights_interval,
                wikibigedit_checkpoint_eval_sample_ratio=wikibigedit_checkpoint_eval_sample_ratio,
                wikibigedit_checkpoint_eval_sample_seed=wikibigedit_checkpoint_eval_sample_seed,
                downstream_eval_steps=downstream_eval_steps,
                after_edits=len(edited_records),
                cnt=cnt,
                checkpoint_eval_count=checkpoint_eval_count,
                last_checkpoint_after=last_checkpoint_after,
                edited_records=edited_records,
                hparams=hparams,
                model=edited_model,
                cache_c=cache_c,
                lyaplock_kwargs=lyaplock_kwargs if alg_name == "LyapLock" else None,
            )

        if memit_prune_base_weights_local is None:
            raise ValueError("Missing MEMIT_prune base weights for finalize step")

        with torch.no_grad():
            upd_matrix = {}
            for k, v in memit_prune_base_weights_local.items():
                current_weight = nethook.get_parameter(edited_model, k)
                v_cuda = v.to(device=current_weight.device, dtype=current_weight.dtype)
                upd_matrix[k] = current_weight - v_cuda
                _, S_orig, _ = torch.svd(v_cuda)
                max_sigma = S_orig.max().item()

                U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                adjusted_S = torch.where(
                    S_upd > max_sigma,
                    torch.log(S_upd)
                    - torch.log(torch.tensor(max_sigma, device=current_weight.device))
                    + max_sigma,
                    S_upd,
                )
                upd_matrix[k] = torch.matmul(
                    U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t())
                )

            for k in upd_matrix:
                original_weight = nethook.get_parameter(edited_model, k)
                adjusted_weight = original_weight + upd_matrix[k]
                original_weight.copy_(adjusted_weight)

        final_after_edits = len(edited_records)
        ckpt_dir = checkpoint_root / f"after_{final_after_edits}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_path = ckpt_dir / "results.jsonl"
        tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")

        print(
            f"Running FINAL (pruned) checkpoint eval after {final_after_edits} edited records -> {out_path}"
        )
        start_eval = time()
        with open(tmp_out_path, "w") as f:
            for eval_idx, record in enumerate(edited_records):
                do_gen = (
                    snips is not None
                    and vec is not None
                    and generation_test_interval is not None
                    and generation_test_interval > 0
                    and (eval_idx % generation_test_interval == 0)
                )
                post_metrics = ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(gen_test_vars if do_gen else [None, None]),
                )
                metrics = {
                    "case_id": record.get("case_id"),
                    "edit_order_idx": eval_idx,
                    "requested_rewrite": record.get("requested_rewrite"),
                    "post": post_metrics,
                }
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        os.replace(tmp_out_path, out_path)
        print("Final checkpoint evaluation took", time() - start_eval)
    else:
        if len(edited_records) != last_checkpoint_after:
            run_checkpoint_eval(after_edits=len(edited_records), model_to_eval=edited_model)

            # Restore original weights
            # with torch.no_grad():
            #     for k, v in weights_copy.items():
            #         nethook.get_parameter(model, k)[...] = v.to("cuda")
        if alg_name in _RESUMABLE_ALGOS:
            _save_resume_state(
                run_dir=run_dir,
                alg_name=alg_name,
                hf_model_name=hf_model_name,
                ds_name=ds_name,
                dataset_size_limit=dataset_size_limit,
                num_edits=num_edits,
                use_cache=use_cache,
                skip_generation_tests=skip_generation_tests,
                generation_test_interval=generation_test_interval,
                edit_log=edit_log,
                checkpoint_eval_interval=checkpoint_eval_interval,
                save_edited_weights_interval=save_edited_weights_interval,
                wikibigedit_checkpoint_eval_sample_ratio=wikibigedit_checkpoint_eval_sample_ratio,
                wikibigedit_checkpoint_eval_sample_seed=wikibigedit_checkpoint_eval_sample_seed,
                downstream_eval_steps=downstream_eval_steps,
                after_edits=len(edited_records),
                cnt=cnt,
                checkpoint_eval_count=checkpoint_eval_count,
                last_checkpoint_after=last_checkpoint_after,
                edited_records=edited_records,
                hparams=hparams,
                model=edited_model,
                cache_c=cache_c,
                lyaplock_kwargs=lyaplock_kwargs if alg_name == "LyapLock" else None,
            )
def get_project(model, tok, layer, hparams, *, get_cov_fn=get_cov_alphaedit):
    force_recompute = False
    cov = get_cov_fn(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","NAS","ENCORE","MEMOIR","MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT", "MEND","NSE", "LyapLock"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--hparam",
        action="append",
        default=[],
        help="Override a hyperparameter from the loaded hparams JSON (repeatable): key=value. "
        "Values are parsed as JSON/Python literal when possible (e.g., --hparam v_lr=0.05 --hparam layers=[10,11]).",
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake", "wikibigedit"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--edit_log",
        dest="edit_log",
        action="store_true",
        help="Create run_xxx/edit_logs and pass edit_log_dir into the editing method (if it supports it).",
    )
    parser.add_argument(
        "--checkpoint_eval_interval",
        type=int,
        default=0,
        help="Run full dataset evaluation every N edit batches (each batch edits --num_edits records). "
        "Each checkpoint re-evaluates all previously edited records in edit order and writes JSONL to "
        "run_xxx/checkpoint_evals/after_{N}/results.jsonl. If 0, only runs the final checkpoint.",
    )
    parser.add_argument(
        "--save_edited_weights_interval",
        type=int,
        default=0,
        help="If >0 and --checkpoint_eval_interval>0, save the edited rewrite-module weights every N checkpoint "
        "evaluations to run_xxx/edited_weights/after_{after_edits}/rewrite_module_weights.pt. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--wikibigedit_checkpoint_eval_sample_ratio",
        type=float,
        default=None,
        help="WikiBigEdit-only: evaluate a deterministic sample of edited records at each checkpoint. "
        "If omitted, defaults to 0.1 when --ds_name=wikibigedit. Set to 1.0 to evaluate all records.",
    )
    parser.add_argument(
        "--wikibigedit_checkpoint_eval_sample_seed",
        type=int,
        default=0,
        help="WikiBigEdit-only: RNG seed for checkpoint sampling (blockwise nested scheme).",
    )
    parser.add_argument(
        "--lyaplock_alpha",
        type=float,
        default=60.0,
        help="LyapLock-only: alpha used to initialize per-layer constraint state.",
    )
    parser.add_argument(
        "--run_dir_override",
        type=str,
        default=None,
        help="If set, write all run artifacts to this directory instead of results/<alg_name>/run_xxx. "
        "Useful for smoke tests and scripted runs.",
    )
    parser.set_defaults(skip_generation_tests=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        downstream_eval_steps=args.downstream_eval_steps,
        edit_log=args.edit_log,
        checkpoint_eval_interval=args.checkpoint_eval_interval,
        save_edited_weights_interval=args.save_edited_weights_interval,
        wikibigedit_checkpoint_eval_sample_ratio=args.wikibigedit_checkpoint_eval_sample_ratio,
        wikibigedit_checkpoint_eval_sample_seed=args.wikibigedit_checkpoint_eval_sample_seed,
        hparam_overrides=args.hparam,
        lyaplock_alpha=args.lyaplock_alpha,
        run_dir_override=args.run_dir_override,
    )
