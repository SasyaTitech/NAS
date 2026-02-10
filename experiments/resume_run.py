import argparse
import json
import os
import importlib
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def _resolve_path(run_dir: Path, rel_or_abs) -> Path:
    if not rel_or_abs:
        raise ValueError("Missing path in resume state")
    path = Path(rel_or_abs)
    return path if path.is_absolute() else (run_dir / path)


def _maybe_chdir(state: dict, run_dir: Path) -> None:
    saved_cwd = state.get("cwd")
    candidates = []
    if saved_cwd:
        candidates.append(Path(saved_cwd))
    for parent in [run_dir] + list(run_dir.parents):
        if (parent / "globals.yml").exists():
            candidates.append(parent)
            break
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            if Path.cwd().resolve() != candidate.resolve():
                os.chdir(candidate)
                print(f"Changed working directory to {candidate}")
            return


_CONTEXT_TEMPLATE_MODULES = {
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

_CACHE_C_ALGOS = {"AlphaEdit", "NAS", "MEMIT_seq", "NSE"}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resume an existing editing run by loading the saved end-of-run state "
            "(edited weights, optional cache_c, optional context templates) and "
            "continuing for additional cases in the same run directory."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to an existing run directory (e.g. results/AlphaEdit/run_008).",
    )
    parser.add_argument(
        "--add_cases",
        type=int,
        required=True,
        help="Number of additional cases to run beyond the saved after_edits.",
    )
    args = parser.parse_args()

    if args.add_cases <= 0:
        raise ValueError("--add_cases must be > 0")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    resume_dir = run_dir / "resume"
    state_path = resume_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing resume state at {state_path}. "
            "Run experiments.evaluate once to generate run_dir/resume/*."
        )

    state = _load_json(state_path)
    alg_name = state.get("algo")
    if not alg_name:
        raise ValueError("Missing algo in resume state.json")
    if alg_name not in {
        "AlphaEdit",
        "NAS",
        "ENCORE",
        "LyapLock",
        "ROME",
        "MEMIT",
        "MEMIT_prune",
        "MEMIT_rect",
        "FT",
        "MEMIT_seq",
        "NSE",
        "UltraEdit",
    }:
        raise ValueError(
            f"Unsupported algo {alg_name!r} in state.json; "
            "supported: AlphaEdit, NAS, ENCORE, LyapLock, ROME, MEMIT, MEMIT_prune, MEMIT_rect, FT, MEMIT_seq, NSE, UltraEdit"
        )

    params_path = run_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}; cannot resume without saved hparams")

    _maybe_chdir(state, run_dir)

    # Delay imports until after we potentially change CWD; util.globals reads globals.yml from CWD.
    import experiments.evaluate as eval_module
    eval_main = eval_module.main
    from util import nethook

    progress = state.get("progress", {})
    after_edits = int(progress.get("after_edits", 0))
    cnt = int(progress.get("cnt", 0))
    checkpoint_eval_count = int(progress.get("checkpoint_eval_count", 0))
    last_checkpoint_after = int(progress.get("last_checkpoint_after", 0))
    edited_case_ids = progress.get("edited_case_ids")

    hf_model_name = state.get("hf_model_name")
    if not hf_model_name:
        raise ValueError("Missing hf_model_name in resume state.json")

    ds_name = state.get("ds_name")
    if not ds_name:
        raise ValueError("Missing ds_name in resume state.json")

    num_edits = int(state.get("num_edits", 1))
    use_cache = bool(state.get("use_cache", False))
    skip_generation_tests = bool(state.get("skip_generation_tests", False))
    generation_test_interval = int(state.get("generation_test_interval", 1))
    edit_log = bool(state.get("edit_log", False))
    checkpoint_eval_interval = int(state.get("checkpoint_eval_interval", 0))
    save_edited_weights_interval = int(state.get("save_edited_weights_interval", 0))
    wikibigedit_checkpoint_eval_sample_ratio = state.get(
        "wikibigedit_checkpoint_eval_sample_ratio"
    )
    wikibigedit_checkpoint_eval_sample_seed = int(
        state.get("wikibigedit_checkpoint_eval_sample_seed", 0)
    )
    downstream_eval_steps = int(state.get("downstream_eval_steps", 0))

    new_dataset_size_limit = after_edits + int(args.add_cases)
    print(
        f"Resuming {alg_name} from after_edits={after_edits} with add_cases={args.add_cases} "
        f"-> dataset_size_limit={new_dataset_size_limit}"
    )

    # Load model/tokenizer in the same way as experiments/evaluate.py for consistency.
    print(f"Loading model/tokenizer from {hf_model_name}")
    model = AutoModelForCausalLM.from_pretrained(hf_model_name).cuda()
    tok = AutoTokenizer.from_pretrained(hf_model_name)
    tok.pad_token = tok.eos_token

    # Restore context templates (avoid re-sampling templates, which is stochastic).
    files = state.get("files", {}) if isinstance(state.get("files"), dict) else {}
    context_templates_rel = files.get("context_templates")
    context_templates_path = (
        _resolve_path(run_dir, context_templates_rel)
        if context_templates_rel
        else (resume_dir / "context_templates.json")
    )
    module_path = _CONTEXT_TEMPLATE_MODULES.get(alg_name)
    if module_path:
        if context_templates_path.exists():
            templates = _load_json(context_templates_path)
            module = importlib.import_module(module_path)
            setattr(module, "CONTEXT_TEMPLATES_CACHE", templates)
            print(f"Loaded context templates from {context_templates_path}")
        else:
            print(
                f"Warning: {context_templates_path} is missing; {alg_name} will regenerate templates "
                "and results may diverge."
            )

    if alg_name in {"AlphaEdit", "NAS"}:
        projection_rel = files.get("projection_matrix")
        default_state_dir = "alphaedit_state" if alg_name == "AlphaEdit" else "nas_state"
        projection_path = (
            _resolve_path(run_dir, projection_rel)
            if projection_rel
            else (run_dir / default_state_dir / "null_space_project.pt")
        )
        if not projection_path.exists():
            try:
                params_class, _ = eval_module.ALG_DICT[alg_name]
                hparams = params_class.from_json(params_path)
                global_projection_path = eval_module._global_projection_cache_path(
                    hf_model_name=str(hf_model_name),
                    hparams=hparams,
                    namespace="alphaedit_family",
                )
                legacy_paths = eval_module._legacy_global_projection_cache_paths(
                    hf_model_name=str(hf_model_name),
                    hparams=hparams,
                )
                hit_path = next(
                    (p for p in [global_projection_path, *legacy_paths] if p.exists()),
                    None,
                )
                if hit_path is not None:
                    print(f"Will load global projection matrix from {hit_path}")
                else:
                    print(
                        f"Warning: missing projection matrix at {projection_path} and no global cache hit; "
                        f"{alg_name} will recompute it, which can be expensive."
                    )
            except Exception:
                print(
                    f"Warning: {projection_path} is missing; {alg_name} may recompute the projection matrix, "
                    "which can be expensive."
                )

    # Restore edited weights from the end-of-run snapshot.
    weights_rel = files.get("rewrite_module_weights")
    weights_path = (
        _resolve_path(run_dir, weights_rel)
        if weights_rel
        else (resume_dir / "rewrite_module_weights.pt")
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing edited weights at {weights_path}")
    weights_blob = torch.load(weights_path, map_location="cpu")
    weights = weights_blob.get("weights", weights_blob)
    memit_prune_base_weights = None
    if alg_name == "MEMIT_prune":
        memit_prune_base_weights = {}
        with torch.no_grad():
            for name in weights.keys():
                param = nethook.get_parameter(model, name)
                memit_prune_base_weights[name] = param.detach().cpu()
        print(f"Captured MEMIT_prune base weights ({len(memit_prune_base_weights)} tensors)")
    print(f"Loading {len(weights)} edited parameter tensors from {weights_path}")
    with torch.no_grad():
        for name, tensor in weights.items():
            param = nethook.get_parameter(model, name)
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))

    if alg_name == "UltraEdit":
        rng_rel = files.get("rng_state")
        rng_path = _resolve_path(run_dir, rng_rel) if rng_rel else (resume_dir / "rng_state.pt")
        if rng_path.exists():
            rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
            try:
                if rng_state.get("torch") is not None:
                    torch.random.set_rng_state(rng_state["torch"])
                cuda_state = rng_state.get("cuda")
                if cuda_state is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(cuda_state)
                    except Exception as e:
                        print(f"Warning: failed to restore CUDA RNG state from {rng_path}: {e}")
                if rng_state.get("python") is not None:
                    random.setstate(rng_state["python"])
                if rng_state.get("numpy") is not None:
                    np.random.set_state(rng_state["numpy"])
                print(f"Restored RNG state from {rng_path}")
            except Exception as e:
                print(f"Warning: failed to restore RNG state from {rng_path}: {e}")
        else:
            print(f"Warning: missing RNG state at {rng_path}; UltraEdit resume may diverge.")

        ultra_rel = files.get("ultraedit_executor_state")
        ultra_state_path = (
            _resolve_path(run_dir, ultra_rel)
            if ultra_rel
            else (resume_dir / "ultraedit_executor_state.pt")
        )
        if ultra_state_path.exists():
            from baselines.ultraedit import UltraEditHyperParams

            ultra_state = torch.load(ultra_state_path, map_location="cpu", weights_only=False)
            hparams = UltraEditHyperParams.from_json(params_path)
            _, apply_algo = eval_module.ALG_DICT["UltraEdit"]
            executor = getattr(apply_algo, "__self__", None)
            if executor is None or not hasattr(executor, "load_state"):
                print(
                    "Warning: UltraEdit executor does not support load_state(); "
                    "resume will proceed without restoring normalizer state."
                )
            else:
                executor.load_state(ultra_state, model=model, tok=tok, hparams=hparams)
                print(f"Loaded UltraEdit executor state from {ultra_state_path}")
        else:
            print(
                f"Warning: missing UltraEdit executor state at {ultra_state_path}; "
                "resume will proceed without restoring normalizer state."
            )

    # Restore sequential statistics for algorithms that require them.
    cache_c = None
    if alg_name in _CACHE_C_ALGOS:
        cache_c_rel = files.get("cache_c")
        cache_c_path = (
            _resolve_path(run_dir, cache_c_rel) if cache_c_rel else (resume_dir / "cache_c.pt")
        )
        if not cache_c_path.exists():
            raise FileNotFoundError(f"Missing cache_c at {cache_c_path}")
        cache_c = torch.load(cache_c_path, map_location="cpu")
        print(f"Loaded cache_c from {cache_c_path}")

    if alg_name == "ENCORE":
        seq_rel = files.get("encore_seq_cache")
        seq_cache_path = (
            _resolve_path(run_dir, seq_rel)
            if seq_rel
            else (resume_dir / "encore_seq_cache.pt")
        )
        if seq_cache_path.exists():
            seq_cache = torch.load(seq_cache_path, map_location="cpu", weights_only=False)
            module = importlib.import_module("baselines.encore.encore_main")
            setattr(module, "SEQ_CACHE", seq_cache)
            print(f"Loaded ENCORE SEQ_CACHE from {seq_cache_path}")
        else:
            print(
                f"Warning: missing ENCORE SEQ_CACHE at {seq_cache_path}; "
                "resume will proceed but results may diverge."
            )

    lyaplock_kwargs = None
    if alg_name == "LyapLock":
        state_rel = files.get("lyaplock_state")
        lyaplock_state_path = (
            _resolve_path(run_dir, state_rel)
            if state_rel
            else (resume_dir / "lyaplock_state.pt")
        )
        if not lyaplock_state_path.exists():
            raise FileNotFoundError(f"Missing LyapLock state at {lyaplock_state_path}")
        lyaplock_kwargs = torch.load(lyaplock_state_path, map_location="cpu", weights_only=False)
        print(f"Loaded LyapLock state from {lyaplock_state_path}")

    eval_main(
        alg_name,
        model_name=(model, tok),
        hparams_fname="params.json",
        ds_name=ds_name,
        dataset_size_limit=new_dataset_size_limit,
        continue_from_run=None,
        skip_generation_tests=skip_generation_tests,
        generation_test_interval=generation_test_interval,
        dir_name=str(run_dir.parent.name),
        num_edits=num_edits,
        use_cache=use_cache,
        downstream_eval_steps=downstream_eval_steps,
        edit_log=edit_log,
        checkpoint_eval_interval=checkpoint_eval_interval,
        save_edited_weights_interval=save_edited_weights_interval,
        wikibigedit_checkpoint_eval_sample_ratio=wikibigedit_checkpoint_eval_sample_ratio,
        wikibigedit_checkpoint_eval_sample_seed=wikibigedit_checkpoint_eval_sample_seed,
        hparam_overrides=None,
        resume_state={
            "after_edits": after_edits,
            "cnt": cnt,
            "checkpoint_eval_count": checkpoint_eval_count,
            "last_checkpoint_after": last_checkpoint_after,
            "edited_case_ids": edited_case_ids,
            **({"cache_c": cache_c} if cache_c is not None else {}),
            **({"lyaplock_kwargs": lyaplock_kwargs} if lyaplock_kwargs is not None else {}),
        },
        run_dir_override=str(run_dir),
        memit_prune_base_weights=memit_prune_base_weights,
    )


if __name__ == "__main__":
    main()
