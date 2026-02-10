import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .NAS_hparams import NASHyperParams

from prompt_templates import fill_subject

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_NAS_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NASHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    edit_log_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, torch.Tensor]:
    """
    Executes the NAS update algorithm for the specified update.
    Invariant: model at beginning of function == model at end of function
    """

    hparams.use_nas = True

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"NAS request sample: "
            f"[{fill_subject(request['prompt'], request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    case_logs = None
    layer_logs = None
    if edit_log_dir is not None:
        case_logs = [
            {
                "case_id": (
                    request.get("case_id").item()
                    if isinstance(request.get("case_id"), np.generic)
                    else request.get("case_id")
                ),
                "residual_pre_mlp_norm": None,
                "target_init_norm": None,
                "v0_norm": None,
                "delta_norm": None,
                "v_star_norm_orig": None,
                "v_star_norm_real": None,
                "target_norm": None,
                "k_norm": {},
            }
            for request in requests
        ]
        layer_logs = []

    for request_idx, request in enumerate(requests):
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            if case_logs is None:
                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
            else:
                cur_z, z_debug = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                    return_debug=True,
                )
                case_logs[request_idx].update(z_debug)

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"Cached k/v pair at {cache_fname}")
        elif case_logs is not None:
            case_logs[request_idx]["target_norm"] = torch.linalg.norm(z_list[-1]).item()
            case_logs[request_idx]["from_cache"] = True
    zs = torch.stack(z_list, dim=1)

    for layer_idx, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
        if case_logs is not None:
            k_norms = torch.linalg.norm(layer_ks, dim=0).tolist()
            for request_idx, k_norm in enumerate(k_norms):
                case_logs[request_idx]["k_norm"][str(layer)] = float(k_norm)

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - layer_idx)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
            P[layer_idx, :, :].cuda()
            @ (layer_ks @ layer_ks.T + cache_c[layer_idx, :, :].cuda())
            + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
            P[layer_idx, :, :].cuda() @ layer_ks @ resid.T,
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        orig_norm = torch.linalg.norm(weights[weight_name]).item()
        upd_norm = torch.linalg.norm(upd_matrix).item()
        print("orig norm", orig_norm)
        print("upd norm", upd_norm)
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
            if layer_logs is not None:
                updated_norm = torch.linalg.norm(weights[weight_name]).item()
                layer_logs.append(
                    {
                        "layer": int(layer),
                        "weight_name": weight_name,
                        "orig_norm": float(orig_norm),
                        "upd_norm": float(upd_norm),
                        "updated_norm": float(updated_norm),
                    }
                )

        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    for layer_idx, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[layer_idx, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    if edit_log_dir is not None:
        edit_log_path = Path(edit_log_dir) / "edits.jsonl"
        edit_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_record = {
            "algo": "NAS",
            "cases": case_logs,
            "layers": layer_logs,
        }
        with open(edit_log_path, "a") as f:
            f.write(json.dumps(log_record) + "\n")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        "Update matrix computed by NAS does not match original weight shape. "
        "Check for bugs in the code?"
    )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
