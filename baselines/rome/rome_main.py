import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

from prompt_templates import fill_subject

CONTEXT_TEMPLATES_CACHE = None
_LAST_UPD_MATRIX_NORM_BY_MODEL_WEIGHT: Dict[Tuple[str, str], float] = {}


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    edit_log_dir = kwargs.get("edit_log_dir")
    deltas = execute_rome(model, tok, request, hparams, edit_log_dir=edit_log_dir)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    edit_log_dir: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROME algorithm for the update: "
        f"[{fill_subject(request['prompt'], request['subject'])}] -> [{request['target_new']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    case_logs: Optional[List[Dict[str, Any]]] = None
    layer_logs: Optional[List[Dict[str, Any]]] = None
    last_case_debug: Optional[Dict[str, float]] = None
    if edit_log_dir is not None:
        case_id = request.get("case_id")
        if isinstance(case_id, np.generic):
            case_id = case_id.item()
        case_logs = [{"case_id": case_id}]
        layer_logs = []

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        if edit_log_dir is None:
            right_vector: torch.Tensor = compute_v(
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                get_context_templates(model, tok, hparams.context_template_length_params),
            )
        else:
            right_vector, case_debug = compute_v(
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                get_context_templates(model, tok, hparams.context_template_length_params),
                return_debug=True,
            )
            last_case_debug = case_debug
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            model_id = str(getattr(model.config, "_name_or_path", "unknown"))
            upd_norm = torch.linalg.norm(upd_matrix).item()
            upd_norm_key = (model_id, weight_name)
            prev_upd_norm = _LAST_UPD_MATRIX_NORM_BY_MODEL_WEIGHT.get(upd_norm_key)
            skip_update = (
                prev_upd_norm is not None
                and prev_upd_norm > 0
                and upd_norm >= 10 * prev_upd_norm
            )

            orig_norm = None
            if layer_logs is not None:
                orig_norm = torch.linalg.norm(weights[weight_name]).item()

            if skip_update:
                print(
                    f"Skip update: ||ΔW|| {upd_norm:.4f} >= 5x last {prev_upd_norm:.4f} "
                    f"for {weight_name}"
                )
                if layer_logs is not None:
                    layer_logs.append(
                        {
                            "layer": int(layer),
                            "weight_name": weight_name,
                            "orig_norm": float(orig_norm),
                            "upd_norm": float(upd_norm),
                            "updated_norm": float(orig_norm),
                        }
                    )
                continue

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
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
            _LAST_UPD_MATRIX_NORM_BY_MODEL_WEIGHT[upd_norm_key] = float(upd_norm)

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    if edit_log_dir is not None:
        if case_logs is not None and last_case_debug is not None:
            case_logs[0].update(last_case_debug)
        edit_log_path = Path(edit_log_dir) / "edits.jsonl"
        edit_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_record = {
            "algo": "ROME",
            "cases": case_logs,
            "layers": layer_logs,
        }
        with open(edit_log_path, "a") as f:
            f.write(json.dumps(log_record) + "\n")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
