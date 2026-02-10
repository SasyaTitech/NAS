import argparse
import json
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MultiCounterFactDataset
from baselines.memit import MEMITHyperParams
from baselines.memit.compute_z import (
    compute_z,
    find_fact_lookup_idx,
    get_module_input_output_at_words,
)
from baselines.memit.memit_main import apply_memit_to_model, get_context_templates
from util import nethook
from util.globals import DATA_DIR, HPARAMS_DIR


@contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        yield


def _jsonable_case_id(case_id: Any) -> Any:
    if isinstance(case_id, np.generic):
        return case_id.item()
    return case_id


def _normalize_request(request: Dict[str, Any]) -> Dict[str, Any]:
    request = dict(request)
    target_new = dict(request["target_new"])
    target_str = target_new.get("str", "")
    if target_str and not target_str.startswith(" "):
        target_new["str"] = " " + target_str
    request["target_new"] = target_new
    return request


def _load_hparams(hparams_spec: str) -> Tuple[Path, MEMITHyperParams]:
    hparams_path = Path(hparams_spec)
    if not hparams_path.exists():
        hparams_path = HPARAMS_DIR / "MEMIT" / hparams_spec
    return hparams_path, MEMITHyperParams.from_json(hparams_path)


def _create_run_dir(out_root: str, dir_name: str, run_name: Optional[str]) -> Path:
    root = Path(out_root) / dir_name
    root.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_dir = root / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    run_ids = [
        int(p.name.split("_")[-1])
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("run_") and p.name.split("_")[-1].isnumeric()
    ]
    run_id = 0 if not run_ids else max(run_ids) + 1
    run_dir = root / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _dtype_from_str(spec: Optional[str]) -> Optional[torch.dtype]:
    if spec is None:
        return None
    spec = spec.lower()
    if spec == "float16":
        return torch.float16
    if spec == "bfloat16":
        return torch.bfloat16
    if spec == "float32":
        return torch.float32
    raise ValueError(f"Unsupported --dtype {spec!r}; expected float16|bfloat16|float32")


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


def _compute_probe_norms_batched(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict[str, Any]],
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[List[str]],
) -> List[Dict[str, float]]:
    """
    Batched version of the compute_z optimization for *probe-only* metrics.
    Returns per-request {"v0_norm", "v_star_norm_orig"} without editing weights.
    """

    if not requests:
        return []

    device = next(model.parameters()).device

    lm_w = nethook.get_module(model, f"{hparams.lm_head_module}").weight.T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # Prepare per-request prompt templates and targets.
    kl_prompt_templates = ["{} is a"]
    rewriting_prompts_by_req: List[List[str]] = []
    target_ids_by_req: List[torch.Tensor] = []
    probe_prompt_templates: List[str] = []

    for request in requests:
        target_ids = tok(request["target_new"]["str"], return_tensors="pt")["input_ids"][0].to(device)
        if target_ids.numel() == 0:
            raise ValueError("Empty target tokenization for request")
        if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
            target_ids = target_ids[1:]
        if target_ids.numel() == 0:
            raise ValueError("Target tokenization becomes empty after stripping BOS/UNK")

        rewriting_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ]
        if not rewriting_prompts:
            raise ValueError("context_templates produced no rewriting prompts")
        rewriting_prompts_by_req.append(rewriting_prompts)
        target_ids_by_req.append(target_ids)
        probe_prompt_templates.append(rewriting_prompts[0])

    rewrite_prompts_per_req = len(rewriting_prompts_by_req[0])
    if any(len(x) != rewrite_prompts_per_req for x in rewriting_prompts_by_req):
        raise ValueError("Mismatched rewriting prompt counts across requests")

    num_requests = len(requests)
    num_rewrite = num_requests * rewrite_prompts_per_req
    num_kl = num_requests * len(kl_prompt_templates)
    total_seqs = num_rewrite + num_kl

    rewrite_templates = [t for req_prompts in rewriting_prompts_by_req for t in req_prompts]
    kl_templates = kl_prompt_templates * num_requests
    all_templates = rewrite_templates + kl_templates

    rewrite_subjects = [req["subject"] for req in requests for _ in range(rewrite_prompts_per_req)]
    kl_subjects = [req["subject"] for req in requests for _ in range(len(kl_prompt_templates))]
    all_subjects = rewrite_subjects + kl_subjects

    input_texts = [tmpl.format(subj) for tmpl, subj in zip(all_templates, all_subjects)]
    input_tok = tok(input_texts, return_tensors="pt", padding=True).to(device)

    # Targets for rewriting sequences only (rewrite first, KL last).
    seq_len = int(input_tok["input_ids"].shape[1])
    rewriting_targets = torch.full((num_rewrite, seq_len), -100, device=device, dtype=torch.long)
    target_lens = torch.empty((num_rewrite,), device=device, dtype=torch.float32)
    for seq_idx in range(num_rewrite):
        req_idx = seq_idx // rewrite_prompts_per_req
        tgt_ids = target_ids_by_req[req_idx]
        ex_len = int(input_tok["attention_mask"][seq_idx].sum().item())
        if ex_len < int(tgt_ids.numel()):
            raise ValueError("Tokenized prompt shorter than target tokens")
        rewriting_targets[seq_idx, ex_len - int(tgt_ids.numel()) : ex_len] = tgt_ids
        target_lens[seq_idx] = float(tgt_ids.numel())

    # Lookup indices for all sequences (rewrite + KL).
    lookup_idxs = [
        find_fact_lookup_idx(tmpl, subj, tok, hparams.fact_token, verbose=False)
        for tmpl, subj in zip(all_templates, all_subjects)
    ]
    lookup_idxs_t = torch.tensor(lookup_idxs, device=device, dtype=torch.long)

    # Sequence -> request mapping (delta is per-request, shared across its prompts).
    seq_to_req = torch.empty((total_seqs,), device=device, dtype=torch.long)
    seq_to_req[:num_rewrite] = torch.arange(num_requests, device=device).repeat_interleave(
        rewrite_prompts_per_req
    )
    seq_to_req[num_rewrite:] = torch.arange(num_requests, device=device).repeat_interleave(
        len(kl_prompt_templates)
    )

    hidden_size = _get_hidden_size(model)
    delta = torch.zeros((num_requests, hidden_size), requires_grad=True, device=device)
    target_init: Optional[torch.Tensor] = None
    kl_distr_init: Optional[torch.Tensor] = None

    loss_layer = max(hparams.v_loss_layer, layer)
    rewrite_layer_name = hparams.layer_module_tmp.format(layer)
    loss_layer_name = hparams.layer_module_tmp.format(loss_layer)
    trace_layers = [loss_layer_name] if loss_layer_name == rewrite_layer_name else [loss_layer_name, rewrite_layer_name]

    first_rewrite_seq_idxs = torch.arange(
        0, num_rewrite, step=rewrite_prompts_per_req, device=device, dtype=torch.long
    )
    first_lookup_idxs = lookup_idxs_t[first_rewrite_seq_idxs]
    batch_idxs = torch.arange(total_seqs, device=device, dtype=torch.long)

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer != rewrite_layer_name:
            return cur_out

        hs = cur_out[0] if isinstance(cur_out, tuple) else cur_out
        # Record initial v* (target_init) on the clean sentence (first rewriting prompt per request).
        if target_init is None:
            if hs.shape[0] == total_seqs:
                target_init = hs[first_rewrite_seq_idxs, first_lookup_idxs].detach().clone()
            elif hs.shape[1] == total_seqs:
                target_init = hs[first_lookup_idxs, first_rewrite_seq_idxs].detach().clone()
            else:
                raise ValueError(f"Unexpected hidden shape {tuple(hs.shape)} at {rewrite_layer_name}")

        # Add per-request delta at each sequence's lookup index.
        if hs.shape[0] == total_seqs:
            hs[batch_idxs, lookup_idxs_t, :] += delta[seq_to_req]
        elif hs.shape[1] == total_seqs:
            hs[lookup_idxs_t, batch_idxs, :] += delta[seq_to_req]
        else:
            raise ValueError(f"Unexpected hidden shape {tuple(hs.shape)} at {rewrite_layer_name}")

        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
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

        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        target_init_norm = target_init.norm(dim=1)
        delta_norm = delta.norm(dim=1)
        weight_decay = hparams.v_weight_decay * (delta_norm / (target_init_norm**2)).mean()

        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        if loss < 5e-2:
            break
        if it == num_steps - 1:
            break

        loss.backward()
        opt.step()

        # Project each request's delta within its L2 ball.
        with torch.no_grad():
            target_init_norm = target_init.norm(dim=1)
            max_norm = hparams.clamp_norm_factor * target_init_norm
            delta_norm = delta.norm(dim=1)
            scale = torch.minimum(torch.ones_like(delta_norm), max_norm / (delta_norm + eps))
            delta.mul_(scale.unsqueeze(1))

    _, v0 = get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=probe_prompt_templates,
        words=[r["subject"] for r in requests],
        module_template=hparams.mlp_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    with torch.no_grad():
        v0_norms = v0.norm(dim=1).float().cpu().tolist()
        v_star_norms = (v0 + delta).norm(dim=1).float().cpu().tolist()

    return [
        {"v0_norm": float(v0_norms[i]), "v_star_norm_orig": float(v_star_norms[i])}
        for i in range(num_requests)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hparams", default="Llama3-8B-test.json")
    p.add_argument("--model_name", default=None)
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--num_records", type=int, default=2000)
    p.add_argument("--num_edits", type=int, default=500)
    p.add_argument("--num_holdout", type=int, default=1500)
    p.add_argument("--edit_every", type=int, default=20)
    p.add_argument("--num_probe", type=int, default=200)
    p.add_argument("--probe_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--dir_name", default="memit_norm_probe")
    p.add_argument("--out_root", default="probe_results")
    p.add_argument("--run_name", default=None)
    p.add_argument("--verbose_probe", action="store_true")

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (compute_z hardcodes device='cuda').")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.num_records <= 0:
        raise ValueError("--num_records must be > 0")
    if args.num_edits < 0:
        raise ValueError("--num_edits must be >= 0")
    if args.num_holdout < 0:
        raise ValueError("--num_holdout must be >= 0")
    if args.num_edits + args.num_holdout != args.num_records:
        raise ValueError("--num_edits + --num_holdout must equal --num_records")
    if args.num_probe < 0:
        raise ValueError("--num_probe must be >= 0")
    if args.probe_batch_size <= 0:
        raise ValueError("--probe_batch_size must be > 0")
    if args.edit_every <= 0:
        raise ValueError("--edit_every must be > 0")

    hparams_path, hparams = _load_hparams(args.hparams)
    model_name = args.model_name or hparams.model_name
    dtype = _dtype_from_str(args.dtype)

    print(f"Loading model {model_name!r}")
    model_kwargs: Dict[str, Any] = {"trust_remote_code": bool(args.trust_remote_code)}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).cuda().eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(args.trust_remote_code))
    tok.pad_token = tok.eos_token

    print("Loading MultiCounterFact")
    ds = MultiCounterFactDataset(DATA_DIR, size=args.num_records)
    records: List[Dict[str, Any]] = ds.data

    edit_records = records[: args.num_edits]
    holdout_records = records[args.num_edits : args.num_edits + args.num_holdout]
    if args.num_probe > len(holdout_records):
        raise ValueError("--num_probe cannot exceed holdout size")

    rng = random.Random(args.seed)
    probe_rel_idxs = rng.sample(range(len(holdout_records)), k=args.num_probe)
    probe_records = [holdout_records[i] for i in probe_rel_idxs]
    probe_requests = [
        _normalize_request(
            {
                "case_id": _jsonable_case_id(r.get("case_id")),
                **(
                    r["requested_rewrite"][0]
                    if isinstance(r["requested_rewrite"], list)
                    else r["requested_rewrite"]
                ),
            }
        )
        for r in probe_records
    ]

    z_layer = hparams.layers[-1]
    v_weight_name = f"{hparams.rewrite_module_tmp.format(z_layer)}.weight"

    run_dir = _create_run_dir(args.out_root, args.dir_name, args.run_name)
    out_jsonl = run_dir / "probe_norms.jsonl"
    cfg_path = run_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "out_root": str(Path(args.out_root)),
                "model_name": model_name,
                "hparams_path": str(hparams_path),
                "hparams": asdict(hparams),
                "dataset": {
                    "name": "multi_counterfact",
                    "num_records": args.num_records,
                    "num_edits": args.num_edits,
                    "num_holdout": args.num_holdout,
                },
                "edit_every": args.edit_every,
                "probe": {
                    "num_probe": args.num_probe,
                    "probe_batch_size": args.probe_batch_size,
                    "seed": args.seed,
                    "holdout_relative_indices": probe_rel_idxs,
                    "case_ids": [_jsonable_case_id(r.get("case_id")) for r in probe_records],
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Writing probe logs -> {out_jsonl}")
    print(f"Writing config -> {cfg_path}")

    context_templates = get_context_templates(model, tok)

    def log_checkpoint(*, after_edits: int):
        w = nethook.get_parameter(model, v_weight_name)
        weight_norm = float(torch.linalg.norm(w).item())

        samples = []
        if args.probe_batch_size == 1:
            suppress = not args.verbose_probe
            with _suppress_output(suppress):
                for idx, request in enumerate(probe_requests):
                    _, debug = compute_z(
                        model,
                        tok,
                        request,
                        hparams,
                        z_layer,
                        context_templates,
                        return_debug=True,
                    )
                    samples.append(
                        {
                            "case_id": request.get("case_id"),
                            "v0_norm": float(debug["v0_norm"]),
                            "v_star_norm_orig": float(debug["v_star_norm_orig"]),
                        }
                    )
                    if (idx + 1) % 20 == 0:
                        torch.cuda.empty_cache()
        else:
            start = 0
            while start < len(probe_requests):
                cur_bs = min(args.probe_batch_size, len(probe_requests) - start)
                batch = probe_requests[start : start + cur_bs]
                try:
                    batch_debug = _compute_probe_norms_batched(
                        model=model,
                        tok=tok,
                        requests=batch,
                        hparams=hparams,
                        layer=z_layer,
                        context_templates=context_templates,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if cur_bs == 1:
                        raise
                    args.probe_batch_size = max(1, cur_bs // 2)
                    print(f"OOM during probe; reducing --probe_batch_size to {args.probe_batch_size}")
                    continue

                for request, debug in zip(batch, batch_debug):
                    samples.append(
                        {
                            "case_id": request.get("case_id"),
                            "v0_norm": float(debug["v0_norm"]),
                            "v_star_norm_orig": float(debug["v_star_norm_orig"]),
                        }
                    )
                start += cur_bs
                torch.cuda.empty_cache()

        record = {
            "after_edits": int(after_edits),
            "v_layer": int(z_layer),
            "weight_name": v_weight_name,
            "weight_norm": weight_norm,
            "samples": samples,
        }
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for edit_idx, record in enumerate(edit_records):
        rewrite = record["requested_rewrite"]
        rewrite = rewrite[0] if isinstance(rewrite, list) else rewrite
        request = _normalize_request({"case_id": _jsonable_case_id(record.get("case_id")), **rewrite})

        print(f"Applying edit {edit_idx + 1}/{len(edit_records)} (case_id={request.get('case_id')})")
        model, _ = apply_memit_to_model(model, tok, [request], hparams, copy=False, return_orig_weights=False)

        after_edits = edit_idx + 1
        if after_edits % args.edit_every == 0:
            print(f"Checkpoint after {after_edits} edits: probing {len(probe_requests)} samples")
            log_checkpoint(after_edits=after_edits)


if __name__ == "__main__":
    main()
