import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_NAS_ROOT = Path(__file__).resolve().parents[1]
if str(_NAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NAS_ROOT))

from baselines.AlphaEdit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from baselines.memit import MEMITHyperParams, apply_memit_to_model
from baselines.memit.memit_rect_main import apply_memit_rect_to_model
from util import nethook
from util.globals import RESULTS_DIR
from util.vstar_stats import get_or_compute_mean_vstar_norm


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _next_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    max_id = -1
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("run_"):
            continue
        try:
            run_id = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        max_id = max(max_id, run_id)
    return root / f"run_{max_id + 1:03d}"


def _load_counterfact_records(counterfact_path: Path) -> list[dict]:
    with open(counterfact_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {counterfact_path}")
    return data


def _select_records(records: list[dict], *, n: int, start: int, seed: int, mode: str) -> list[dict]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if n <= 0:
        raise ValueError("--num_edits must be > 0")
    if start + n > len(records) and mode == "first":
        raise ValueError(f"Requested {n} records at start={start}, but dataset has {len(records)}")

    if mode == "first":
        return records[start : start + n]
    if mode == "random":
        rng = random.Random(seed)
        if n > len(records):
            raise ValueError(f"--num_edits={n} > dataset size {len(records)}")
        return [records[i] for i in rng.sample(range(len(records)), n)]
    raise ValueError(f"Unknown --select_mode={mode!r}")


def _request_from_counterfact_record(rec: dict) -> dict:
    req = rec["requested_rewrite"]
    return {"case_id": rec["case_id"], **req}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_model_and_tok(model_name_or_path: str, *, torch_dtype: str | None):
    dtype = None
    if torch_dtype:
        if torch_dtype == "fp16":
            dtype = torch.float16
        elif torch_dtype == "bf16":
            dtype = torch.bfloat16
        elif torch_dtype == "fp32":
            dtype = torch.float32
        else:
            raise ValueError("--torch_dtype must be one of fp16|bf16|fp32")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype).cuda()
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    tok.pad_token = tok.eos_token
    return model, tok


def _warmup_context_templates(alg_name: str, model, tok) -> None:
    if alg_name in {"MEMIT", "MEMIT_prune"}:
        from baselines.memit.memit_main import get_context_templates  # local cache in module
    elif alg_name == "MEMIT_rect":
        from baselines.memit.memit_rect_main import get_context_templates
    elif alg_name == "AlphaEdit":
        from baselines.AlphaEdit.AlphaEdit_main import get_context_templates
    else:
        return
    _ = get_context_templates(model, tok)


def _prewarm_nas_mean_norm(*, model, tok, hparams, layer: int) -> None:
    if not getattr(hparams, "use_nas", False):
        return
    _ = get_or_compute_mean_vstar_norm(model=model, tok=tok, target_layer=layer, algo_hparams=hparams)


def _alphaedit_init_state(*, model, tok, hparams: AlphaEditHyperParams, shared_state_dir: Path):
    from experiments.evaluate import get_project

    rewrite_module_name = hparams.rewrite_module_tmp.format(hparams.layers[-1])
    rewrite_module = nethook.get_module(model, rewrite_module_name)
    W_out = nethook.get_parameter(model, f"{rewrite_module_name}.weight")

    if hasattr(rewrite_module, "in_features"):
        k_dim = int(rewrite_module.in_features)
    elif rewrite_module.__class__.__name__ == "Conv1D":
        k_dim = int(W_out.shape[0])
    else:
        k_dim = int(W_out.shape[1]) if W_out.ndim == 2 else int(W_out.shape[-1])

    cache_c = torch.zeros((len(hparams.layers), k_dim, k_dim), device="cpu")

    shared_state_dir.mkdir(parents=True, exist_ok=True)
    projection_path = shared_state_dir / "alphaedit_null_space_project.pt"
    if projection_path.exists():
        P = torch.load(projection_path, map_location="cpu")
    else:
        P = torch.zeros((len(hparams.layers), k_dim, k_dim), device="cpu")
        for i, layer in enumerate(hparams.layers):
            P[i, :, :] = get_project(model, tok, layer, hparams)
        torch.save(P, projection_path)
    return cache_c, P


def _memit_prune_finalize(*, edited_model, hparams: MEMITHyperParams, base_weights: dict[str, torch.Tensor]) -> None:
    upd_matrix = {}
    with torch.no_grad():
        for k, v in base_weights.items():
            current_weight = nethook.get_parameter(edited_model, k)
            v_cuda = v.to(device=current_weight.device, dtype=current_weight.dtype)
            upd_matrix[k] = current_weight - v_cuda

            _, S_orig, _ = torch.svd(v_cuda)
            max_sigma = S_orig.max().item()

            U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
            adjusted_S = torch.where(
                S_upd > max_sigma,
                torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device=current_weight.device)) + max_sigma,
                S_upd,
            )
            upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

        for k in upd_matrix:
            original_weight = nethook.get_parameter(edited_model, k)
            original_weight.copy_(original_weight + upd_matrix[k])


def run_one_condition(
    *,
    bench_dir: Path,
    alg_name: str,
    use_nas: bool,
    model_name_or_path: str,
    torch_dtype: str | None,
    hparams_path: Path,
    selected_records: list[dict],
):
    out_dir = bench_dir / f"{alg_name}" / ("nas_on" if use_nas else "nas_off")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = _load_model_and_tok(model_name_or_path, torch_dtype=torch_dtype)

    if alg_name in {"MEMIT", "MEMIT_prune", "MEMIT_rect"}:
        hparams = MEMITHyperParams.from_json(hparams_path)
    elif alg_name == "AlphaEdit":
        hparams = AlphaEditHyperParams.from_json(hparams_path)
    else:
        raise ValueError(f"Unknown algorithm {alg_name}")

    hparams.use_nas = bool(use_nas)

    _warmup_context_templates(alg_name, model, tok)
    _prewarm_nas_mean_norm(model=model, tok=tok, hparams=hparams, layer=hparams.layers[-1])

    alpha_cache_c = None
    alpha_P = None
    if alg_name == "AlphaEdit":
        alpha_cache_c, alpha_P = _alphaedit_init_state(
            model=model, tok=tok, hparams=hparams, shared_state_dir=(bench_dir / "shared_state")
        )

    memit_prune_base_weights = None
    if alg_name == "MEMIT_prune":
        memit_prune_base_weights = {}
        with torch.no_grad():
            for layer in hparams.layers:
                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                memit_prune_base_weights[weight_name] = (
                    nethook.get_parameter(model, weight_name).detach().cpu()
                )

    per_edit = []
    for edit_idx, rec in enumerate(selected_records):
        request = _request_from_counterfact_record(rec)
        _sync()
        t0 = time.perf_counter()

        if alg_name == "AlphaEdit":
            model, alpha_cache_c = apply_AlphaEdit_to_model(
                model,
                tok,
                [request],
                hparams,
                cache_template=None,
                cache_c=alpha_cache_c,
                P=alpha_P,
                edit_log_dir=None,
            )
        elif alg_name in {"MEMIT", "MEMIT_prune"}:
            model, _ = apply_memit_to_model(
                model,
                tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=False,
                cache_template=None,
                edit_log_dir=None,
            )
        elif alg_name == "MEMIT_rect":
            model, _ = apply_memit_rect_to_model(
                model,
                tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=False,
                cache_template=None,
                edit_log_dir=None,
            )
        else:
            raise RuntimeError("unreachable")

        _sync()
        dt = time.perf_counter() - t0
        per_edit.append(
            {
                "edit_idx": edit_idx,
                "case_id": rec.get("case_id"),
                "time_s": float(dt),
            }
        )
        _atomic_write_json(out_dir / "progress_last.json", per_edit[-1])

    prune_finalize_s = None
    if alg_name == "MEMIT_prune":
        _sync()
        t0 = time.perf_counter()
        _memit_prune_finalize(edited_model=model, hparams=hparams, base_weights=memit_prune_base_weights)
        _sync()
        prune_finalize_s = float(time.perf_counter() - t0)

    times = [x["time_s"] for x in per_edit]
    summary = {
        "alg_name": alg_name,
        "use_nas": bool(use_nas),
        "model_name_or_path": model_name_or_path,
        "hparams_path": str(hparams_path),
        "num_edits": len(per_edit),
        "mean_time_s": float(np.mean(times)) if times else None,
        "std_time_s": float(np.std(times)) if times else None,
        "median_time_s": float(np.median(times)) if times else None,
        "min_time_s": float(np.min(times)) if times else None,
        "max_time_s": float(np.max(times)) if times else None,
        "memit_prune_finalize_s": prune_finalize_s,
        "hparams": asdict(hparams),
    }
    _atomic_write_json(out_dir / "summary.json", summary)
    _atomic_write_json(out_dir / "per_edit_time.json", per_edit)
    del model
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--counterfact_path", default=None, type=str)
    parser.add_argument("--num_edits", default=100, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--select_mode", default="first", choices=["first", "random"])
    parser.add_argument("--select_seed", default=0, type=int)
    parser.add_argument("--torch_dtype", default=None, choices=[None, "fp16", "bf16", "fp32"], nargs="?")
    args = parser.parse_args()

    if args.counterfact_path is None:
        alphaedit_root = Path(__file__).resolve().parents[1]
        args.counterfact_path = str(alphaedit_root / "data" / "counterfact.json")

    counterfact_path = Path(args.counterfact_path)
    records = _load_counterfact_records(counterfact_path)
    selected = _select_records(
        records, n=int(args.num_edits), start=int(args.start), seed=int(args.select_seed), mode=args.select_mode
    )

    bench_root = RESULTS_DIR / "SeqEditTimeBench"
    bench_dir = _next_run_dir(bench_root)
    bench_dir.mkdir(parents=True, exist_ok=True)

    selection_info = {
        "counterfact_path": str(counterfact_path),
        "num_edits": int(args.num_edits),
        "start": int(args.start),
        "select_mode": args.select_mode,
        "select_seed": int(args.select_seed),
        "case_ids": [r.get("case_id") for r in selected],
    }
    _atomic_write_json(bench_dir / "selection.json", selection_info)

    conditions = [
        ("MEMIT", False),
        ("MEMIT", True),
        ("MEMIT_prune", False),
        ("MEMIT_prune", True),
        ("MEMIT_rect", False),
        ("MEMIT_rect", True),
        ("AlphaEdit", False),
        ("AlphaEdit", True),
    ]

    summaries = []
    for alg_name, use_nas in conditions:
        if alg_name in {"MEMIT", "MEMIT_prune", "MEMIT_rect"}:
            hparams_path = Path(__file__).resolve().parents[1] / "hparams" / "MEMIT" / "Llama3-8B.json"
        elif alg_name == "AlphaEdit":
            hparams_path = Path(__file__).resolve().parents[1] / "hparams" / "AlphaEdit" / "Llama3-8B.json"
        else:
            raise RuntimeError("unreachable")

        print(f"\n===== Running {alg_name} (use_nas={use_nas}) =====")
        summary = run_one_condition(
            bench_dir=bench_dir,
            alg_name=alg_name,
            use_nas=use_nas,
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            hparams_path=hparams_path,
            selected_records=selected,
        )
        summaries.append(summary)
        _atomic_write_json(bench_dir / "summary_all.json", summaries)

    print(f"Wrote benchmark results to {bench_dir}")


if __name__ == "__main__":
    main()
