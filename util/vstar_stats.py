from __future__ import annotations

import json
import random
import gc
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MultiCounterFactDataset
from baselines.memit import MEMITHyperParams
from baselines.rome import repr_tools
from util import nethook
from util.globals import DATA_DIR, HPARAMS_DIR, STATS_DIR

_VSTAR_MEAN_NORM_CACHE: Dict[Tuple[str, int, int, str, str, float], float] = {}

_NAS_SEED = 0
_NAS_BATCH_SIZE = 16
_NAS_PROGRESS = True
_NAS_OUTLIER_MODE = "drop"
_NAS_OUTLIER_UPPER_QUANTILE = 0.95

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None


def scale_vector_to_norm(x: torch.Tensor, target_norm: float, *, eps: float = 1e-10) -> torch.Tensor:
    if target_norm <= 0:
        return x
    with torch.no_grad():
        denom = x.norm() + eps
        if denom <= eps:
            return x
        return x * (float(target_norm) / denom)


def get_or_compute_mean_vstar_norm(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    target_layer: int,
    algo_hparams: Any,
) -> float:
    nas_restart = int(getattr(algo_hparams, "nas_restart"))
    hparams_name = getattr(algo_hparams, "_hparams_name", None)
    if not hparams_name:
        raise ValueError(
            "NAS requires hyperparameters to be loaded via HyperParams.from_json "
            "(missing attribute `_hparams_name`)."
        )

    model_id = model.config._name_or_path.replace("/", "_")
    resolved_hparams_name = str(hparams_name)
    memit_hparams_path = HPARAMS_DIR / "MEMIT" / resolved_hparams_name
    if not memit_hparams_path.exists():
        # Resumed runs often load algo hparams from run_dir/params.json, so the filename
        # becomes "params.json". Fall back to selecting an appropriate MEMIT hparams file
        # based on the model_name stored in the algo hparams.
        target_model_name = getattr(algo_hparams, "model_name", None)
        if isinstance(target_model_name, str) and target_model_name:
            candidates: List[Path] = []
            for candidate in (HPARAMS_DIR / "MEMIT").glob("*.json"):
                try:
                    with open(candidate, "r") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                if meta.get("model_name") == target_model_name:
                    candidates.append(candidate)
            if candidates:
                non_test = [p for p in candidates if "test" not in p.stem]
                memit_hparams_path = sorted(non_test or candidates)[0]
                resolved_hparams_name = memit_hparams_path.name

    cache_key = (
        model_id,
        int(target_layer),
        nas_restart,
        str(resolved_hparams_name),
        str(_NAS_OUTLIER_MODE),
        float(_NAS_OUTLIER_UPPER_QUANTILE),
    )
    if cache_key in _VSTAR_MEAN_NORM_CACHE:
        return _VSTAR_MEAN_NORM_CACHE[cache_key]

    stats_path = _stats_path_for_model(model_id)
    stats = _load_json(stats_path) or {"schema_version": 2, "model_id": model_id, "layers": {}}
    if "schema_version" not in stats:
        stats["schema_version"] = 2
    layers = stats.setdefault("layers", {})

    layer_key = str(int(target_layer))
    entry = layers.get(layer_key)
    if (
        isinstance(entry, dict)
        and entry.get("nas_restart") == nas_restart
        and entry.get("hparams_name") == str(resolved_hparams_name)
        and isinstance(entry.get("outlier_policy"), dict)
        and entry["outlier_policy"].get("mode") == _NAS_OUTLIER_MODE
        and float(entry["outlier_policy"].get("upper_quantile", -1)) == float(_NAS_OUTLIER_UPPER_QUANTILE)
        and isinstance(entry.get("mean_vstar_norm"), (int, float))
    ):
        mean_norm = float(entry["mean_vstar_norm"])
        _VSTAR_MEAN_NORM_CACHE[cache_key] = mean_norm
        return mean_norm
    if not memit_hparams_path.exists():
        raise FileNotFoundError(
            f"Missing MEMIT hparams file {memit_hparams_path}. "
            "Expected same filename as the current algo hparams, or a MEMIT hparams file with matching model_name."
        )
    memit_hparams = MEMITHyperParams.from_json(memit_hparams_path)

    mean_norm, nas_meta = _compute_mean_vstar_norm_from_multicounterfact(
        model=model,
        tok=tok,
        target_layer=int(target_layer),
        memit_hparams=memit_hparams,
        nas_restart=nas_restart,
        seed=_NAS_SEED,
        batch_size=_NAS_BATCH_SIZE,
        outlier_mode=_NAS_OUTLIER_MODE,
        outlier_upper_quantile=float(_NAS_OUTLIER_UPPER_QUANTILE),
    )

    layers[layer_key] = {
        "mean_vstar_norm": float(mean_norm),
        "nas_restart": nas_restart,
        "hparams_name": str(resolved_hparams_name),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": _NAS_SEED,
        "outlier_policy": {"mode": _NAS_OUTLIER_MODE, "upper_quantile": float(_NAS_OUTLIER_UPPER_QUANTILE)},
        **nas_meta,
        "memit_hparams": {
            k: v
            for k, v in asdict(memit_hparams).items()
            if k
            in {
                "fact_token",
                "v_num_grad_steps",
                "v_lr",
                "v_loss_layer",
                "v_weight_decay",
                "clamp_norm_factor",
                "kl_factor",
                "layer_module_tmp",
                "mlp_module_tmp",
                "ln_f_module",
                "lm_head_module",
            }
        },
        "prompt_mode": "bare",
        "dataset": "multi_counterfact",
    }
    _save_json(stats_path, stats)

    _VSTAR_MEAN_NORM_CACHE[cache_key] = float(mean_norm)
    return float(mean_norm)


def _stats_path_for_model(model_id: str) -> Path:
    return STATS_DIR / f"{model_id}_vstar_norm.json"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


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


def _compute_mean_vstar_norm_from_multicounterfact(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    target_layer: int,
    memit_hparams: MEMITHyperParams,
    nas_restart: int,
    seed: int,
    batch_size: int,
    outlier_mode: str,
    outlier_upper_quantile: float,
) -> Tuple[float, Dict[str, Any]]:
    if nas_restart <= 0:
        raise ValueError("nas_restart must be > 0")
    if outlier_mode != "drop":
        raise ValueError(f"Unsupported outlier_mode={outlier_mode!r}")
    if not (0.0 < float(outlier_upper_quantile) <= 1.0):
        raise ValueError("outlier_upper_quantile must be in (0, 1]")

    ds = MultiCounterFactDataset(DATA_DIR)
    pool: List[Dict[str, Any]] = []
    for record in ds:
        rewrites = record.get("requested_rewrite")
        if isinstance(rewrites, list):
            items = rewrites
        else:
            items = [rewrites]
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
        raise RuntimeError("No valid CounterFact rewrites found for NAS statistics.")

    rng = random.Random(int(seed))
    if nas_restart >= len(pool):
        sample = list(pool)
    else:
        sample = rng.sample(pool, nas_restart)

    initial_batch_size = int(batch_size)
    if initial_batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    cur_batch_size = initial_batch_size
    min_batch_size_used = cur_batch_size
    oom_retries = 0

    pbar = None
    if _NAS_PROGRESS and _tqdm is not None:
        pbar = _tqdm(
            total=nas_restart,
            desc=f"NAS mean ||v*|| (layer {target_layer})",
            unit="sample",
        )
    elif _NAS_PROGRESS:
        print(
            f"NAS mean ||v*||: computing {nas_restart} samples "
            f"(initial_batch_size={initial_batch_size}, layer={target_layer})"
        )

    running_sum = 0.0
    running_count = 0
    all_norms: List[float] = []
    idx = 0
    try:
        while idx < nas_restart:
            bs = min(cur_batch_size, nas_restart - idx)
            batch = sample[idx : idx + bs]

            try:
                batch_norms = _compute_vstar_norms_batched(
                    model=model,
                    tok=tok,
                    requests=batch,
                    hparams=memit_hparams,
                    layer=target_layer,
                )
            except RuntimeError as e:
                if _is_cuda_oom(e) and bs > 1:
                    oom_retries += 1
                    new_bs = max(1, bs // 2)
                    if _NAS_PROGRESS:
                        msg = f"CUDA OOM at batch_size={bs}; retrying with batch_size={new_bs}"
                        if pbar is not None:
                            pbar.write(msg)
                        else:
                            print(msg)
                    cur_batch_size = new_bs
                    min_batch_size_used = min(min_batch_size_used, cur_batch_size)
                    _recover_from_oom()
                    continue
                raise

            batch_sum = float(np.sum(np.asarray(batch_norms, dtype=np.float64)))
            running_sum += batch_sum
            running_count += len(batch_norms)
            all_norms.extend([float(x) for x in batch_norms])
            idx += len(batch_norms)

            raw_mean_so_far = running_sum / max(running_count, 1)
            trim_mean_so_far, drop_so_far, _ = _upper_trimmed_mean(
                all_norms, upper_quantile=float(outlier_upper_quantile)
            )
            if pbar is not None:
                pbar.update(len(batch_norms))
                pbar.set_postfix(
                    mean=f"{trim_mean_so_far:.4f}",
                    raw=f"{raw_mean_so_far:.4f}",
                    drop=f"{drop_so_far}/{running_count}",
                    bs=cur_batch_size,
                    oom=oom_retries,
                )
            elif _NAS_PROGRESS:
                print(
                    f"NAS mean ||v*||: {running_count}/{nas_restart} "
                    f"mean={trim_mean_so_far:.4f} raw={raw_mean_so_far:.4f} "
                    f"drop={drop_so_far}/{running_count} bs={cur_batch_size} oom={oom_retries}"
                )

        raw_mean_norm = float(running_sum / max(running_count, 1))
        mean_norm, drop_count, drop_threshold = _upper_trimmed_mean(
            all_norms, upper_quantile=float(outlier_upper_quantile)
        )
        if _NAS_PROGRESS:
            print(
                f"NAS mean ||v*|| (layer {target_layer}) = {mean_norm:.6f} "
                f"(raw_mean={raw_mean_norm:.6f}, dropped_top={drop_count}/{running_count})"
            )
        meta = {
            "initial_batch_size": initial_batch_size,
            "final_batch_size": cur_batch_size,
            "min_batch_size_used": min_batch_size_used,
            "oom_retries": oom_retries,
            "raw_mean_vstar_norm": raw_mean_norm,
            "outlier_drop_count": int(drop_count),
            "outlier_drop_threshold": float(drop_threshold) if drop_threshold is not None else None,
        }
        return float(mean_norm), meta
    finally:
        if pbar is not None:
            pbar.close()


def _upper_trimmed_mean(values: Sequence[float], *, upper_quantile: float) -> Tuple[float, int, Optional[float]]:
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
        raise ValueError("Trimmed away all samples; lower outlier trimming or increase nas_restart")
    return float(kept.mean()), int(drop_count), float(kept[-1])


def _batched(items: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def _normalize_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = request.get("prompt")
    subject = request.get("subject")
    target_new = request.get("target_new")
    if not isinstance(prompt, str) or not isinstance(subject, str) or not isinstance(target_new, dict):
        return None

    target_str = target_new.get("str", "")
    if not isinstance(target_str, str) or not target_str:
        return None
    if not target_str.startswith(" "):
        target_new = dict(target_new)
        target_new["str"] = " " + target_str

    if "{}" not in prompt and subject in prompt:
        prompt = prompt.replace(subject, "{}")

    return {"prompt": prompt, "subject": subject, "target_new": target_new}


def _get_hidden_size(model: AutoModelForCausalLM) -> int:
    if hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    raise NotImplementedError("Could not infer hidden size from model.config")


def _ensure_batch_first(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if x.shape[0] == batch_size:
        return x
    if x.shape[1] == batch_size:
        return x.transpose(0, 1)
    raise ValueError(f"Unexpected tensor shape {tuple(x.shape)} for batch_size={batch_size}")


def _compute_vstar_norms_batched(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict[str, Any]],
    hparams: MEMITHyperParams,
    layer: int,
) -> List[float]:
    if not requests:
        return []

    device = next(model.parameters()).device

    lm_w = nethook.get_module(model, f"{hparams.lm_head_module}").weight.T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    target_ids_by_req: List[torch.Tensor] = []
    rewrite_templates: List[str] = []
    subjects: List[str] = []
    for request in requests:
        subj = request["subject"]
        target_ids = tok(request["target_new"]["str"], return_tensors="pt")["input_ids"][0].to(device)
        if target_ids.numel() == 0:
            raise ValueError("Empty target tokenization for request")
        if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
            target_ids = target_ids[1:]
        if target_ids.numel() == 0:
            raise ValueError("Target tokenization becomes empty after stripping BOS/UNK")

        rewrite_templates.append(request["prompt"] + tok.decode(target_ids[:-1]))
        target_ids_by_req.append(target_ids)
        subjects.append(subj)

    num_requests = len(requests)
    kl_template = "{} is a"
    all_templates = rewrite_templates + [kl_template] * num_requests
    all_subjects = subjects + subjects

    input_texts = [tmpl.format(subj) for tmpl, subj in zip(all_templates, all_subjects)]
    input_tok = tok(input_texts, return_tensors="pt", padding=True).to(device)

    total_seqs = 2 * num_requests
    num_rewrite = num_requests
    num_kl = num_requests

    seq_len = int(input_tok["input_ids"].shape[1])
    rewriting_targets = torch.full((num_rewrite, seq_len), -100, device=device, dtype=torch.long)
    target_lens = torch.empty((num_rewrite,), device=device, dtype=torch.float32)
    for seq_idx in range(num_rewrite):
        tgt_ids = target_ids_by_req[seq_idx]
        ex_len = int(input_tok["attention_mask"][seq_idx].sum().item())
        if ex_len < int(tgt_ids.numel()):
            raise ValueError("Tokenized prompt shorter than target tokens")
        rewriting_targets[seq_idx, ex_len - int(tgt_ids.numel()) : ex_len] = tgt_ids
        target_lens[seq_idx] = float(tgt_ids.numel())

    lookup_idxs = [
        _find_fact_lookup_idx(tmpl, subj, tok, hparams.fact_token)
        for tmpl, subj in zip(all_templates, all_subjects)
    ]
    lookup_idxs_t = torch.tensor(lookup_idxs, device=device, dtype=torch.long)

    seq_to_req = torch.cat(
        [torch.arange(num_requests, device=device), torch.arange(num_requests, device=device)]
    )

    hidden_size = _get_hidden_size(model)
    delta = torch.zeros((num_requests, hidden_size), requires_grad=True, device=device)
    target_init: Optional[torch.Tensor] = None
    kl_distr_init: Optional[torch.Tensor] = None

    loss_layer = max(int(hparams.v_loss_layer), int(layer))
    rewrite_layer_name = hparams.layer_module_tmp.format(layer)
    loss_layer_name = hparams.layer_module_tmp.format(loss_layer)
    trace_layers = [loss_layer_name] if loss_layer_name == rewrite_layer_name else [loss_layer_name, rewrite_layer_name]

    batch_idxs = torch.arange(total_seqs, device=device, dtype=torch.long)
    rewrite_seq_idxs = torch.arange(num_rewrite, device=device, dtype=torch.long)
    rewrite_lookup_idxs = lookup_idxs_t[rewrite_seq_idxs]

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer != rewrite_layer_name:
            return cur_out

        hs = cur_out[0] if isinstance(cur_out, tuple) else cur_out

        if target_init is None:
            if hs.shape[0] == total_seqs:
                target_init = hs[rewrite_seq_idxs, rewrite_lookup_idxs].detach().clone()
            elif hs.shape[1] == total_seqs:
                target_init = hs[rewrite_lookup_idxs, rewrite_seq_idxs].detach().clone()
            else:
                raise ValueError(f"Unexpected hidden shape {tuple(hs.shape)} at {rewrite_layer_name}")

        if hs.shape[0] == total_seqs:
            hs[batch_idxs, lookup_idxs_t, :] += delta[seq_to_req]
        elif hs.shape[1] == total_seqs:
            hs[lookup_idxs_t, batch_idxs, :] += delta[seq_to_req]
        else:
            raise ValueError(f"Unexpected hidden shape {tuple(hs.shape)} at {rewrite_layer_name}")

        return cur_out

    opt = torch.optim.Adam([delta], lr=float(hparams.v_lr))
    nethook.set_requires_grad(False, model)

    kl_seq_start = num_rewrite
    kl_batch_idxs = torch.arange(kl_seq_start, kl_seq_start + num_kl, device=device, dtype=torch.long)
    eps = 1e-10
    num_steps = int(hparams.v_num_grad_steps)

    for it in range(num_steps):
        opt.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=trace_layers,
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            if logits.shape[0] != total_seqs and logits.shape[1] == total_seqs:
                logits = logits.transpose(0, 1)
            if logits.shape[0] != total_seqs:
                raise ValueError(f"Unexpected logits shape {tuple(logits.shape)} for batch={total_seqs}")

            kl_token_idxs = lookup_idxs_t[kl_batch_idxs]
            kl_logits = logits[kl_batch_idxs, kl_token_idxs, :]
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        if target_init is None or kl_distr_init is None:
            raise RuntimeError("Failed to initialize target_init / kl_distr_init")

        out = tr[loss_layer_name].output
        out = out[0] if isinstance(out, tuple) else out
        out = _ensure_batch_first(out, batch_size=total_seqs)
        full_repr = out[:num_rewrite]

        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device),
            dim=2,
        )
        gather_idx = torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2)
        token_log_probs = torch.gather(log_probs, 2, gather_idx).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(token_log_probs * mask).sum(1) / target_lens
        nll_loss = nll_loss_each.mean()

        kl_loss = float(hparams.kl_factor) * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        target_init_norm = target_init.norm(dim=1)
        delta_norm = delta.norm(dim=1)
        weight_decay = float(hparams.v_weight_decay) * (delta_norm / (target_init_norm**2)).mean()

        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        if loss < 5e-2:
            break
        if it == num_steps - 1:
            break

        loss.backward()
        opt.step()

        with torch.no_grad():
            max_norm = float(hparams.clamp_norm_factor) * target_init_norm
            delta_norm = delta.norm(dim=1)
            scale = torch.minimum(torch.ones_like(delta_norm), max_norm / (delta_norm + eps))
            delta.mul_(scale.unsqueeze(1))

    _, v0 = _get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=rewrite_templates,
        words=subjects,
        module_template=hparams.mlp_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    with torch.no_grad():
        v_star_norms = (v0 + delta).norm(dim=1).float().cpu().tolist()

    return [float(x) for x in v_star_norms]


def _get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("fact_token=last is not supported for NAS statistics.")
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def _find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
) -> int:
    if fact_token_strategy == "last":
        return -1
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        return repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    raise ValueError(f"fact_token={fact_token_strategy} not recognized")
