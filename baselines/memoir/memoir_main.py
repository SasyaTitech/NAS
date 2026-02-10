from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import DATA_DIR

from .memoir_hparams import MEMOIRHyperParams
from .memoir_model import MEMOIRAdapter, MEMOIRWrapper
from .memoir_utils import get_context_templates

from prompt_templates import fill_subject


def _resolve_background_features_path(hparams: MEMOIRHyperParams) -> Path:
    path = Path(hparams.dir_background_features)
    if not path.is_absolute():
        path = (Path(DATA_DIR) / path).resolve()
    return path


def _ensure_background_features(hparams: MEMOIRHyperParams) -> None:
    dst = _resolve_background_features_path(hparams)
    if dst.exists():
        return

    repo_root = Path(__file__).resolve().parents[3]
    src = repo_root / "MEMOIR" / "background_features" / dst.name
    if not src.exists():
        raise FileNotFoundError(
            f"Missing MEMOIR background features at {dst} (and no fallback at {src})."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _normalize_request(request: Dict[str, Any]) -> Dict[str, str]:
    subject = request.get("subject", "")
    prompt = request.get("prompt", "")
    if isinstance(prompt, str) and any(ph in prompt for ph in ("{}", "{subject}", "{0}")):
        prompt = fill_subject(prompt, subject)

    target_new = request.get("target_new")
    if isinstance(target_new, dict):
        target_new = target_new.get("str", "")
    if not isinstance(target_new, str):
        target_new = str(target_new)

    return {"prompt": prompt, "target_new": target_new}


def _tokenize_single_edit(*, prompt: str, target_new: str, tok: AutoTokenizer, device: str, context_templates: List[str]):
    mask_token = -100

    loc_prompt = "<<MEMOIR_IRRELEVANT>>"
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token_id = tok.eos_token_id
    prompt_texts = [templ.format(prompt) for templ in context_templates] + [loc_prompt]
    full_texts = [templ.format(f"{prompt} {target_new}") for templ in context_templates] + [
        loc_prompt
    ]

    prompt_tok = tok(prompt_texts, return_tensors="pt", padding=True, truncation=True)
    tokens = tok(full_texts, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()

    prompt_lens = prompt_tok["attention_mask"].sum(dim=1).to(dtype=torch.long)
    last_prompt_token_loc = (prompt_lens - 1).clamp(min=0)

    for i in range(len(context_templates)):
        tokens["labels"][i, : prompt_lens[i]] = mask_token

    tokens["labels"][tokens["attention_mask"] == 0] = mask_token
    tokens = {k: v.to(device) for k, v in tokens.items()}
    last_prompt_token_loc = last_prompt_token_loc.to(device)
    return tokens, last_prompt_token_loc


def apply_memoir_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMOIRHyperParams,
    copy: bool = False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)

    _ensure_background_features(hparams)
    hparams.dir_background_features = str(_resolve_background_features_path(hparams))

    device = f"cuda:{int(hparams.device)}"
    if hasattr(tok, "padding_side"):
        tok.padding_side = "right"

    length_params = hparams.context_template_length_params or [[5, 5], [10, 5]]
    cache_dir = Path(DATA_DIR) / "memoir" / "context_templates"
    context_templates = get_context_templates(
        model=model,
        tok=tok,
        length_params=length_params,
        device=device,
        cache_dir=cache_dir,
    )

    wrapper = MEMOIRWrapper(hparams, model)

    for request in requests:
        norm = _normalize_request(request)
        tokens, last_prompt_token_loc = _tokenize_single_edit(
            prompt=norm["prompt"],
            target_new=norm["target_new"],
            tok=tok,
            device=device,
            context_templates=context_templates,
        )
        wrapper.edit(tokens, last_prompt_token_loc=last_prompt_token_loc)

    adapter = wrapper.get_adapter_layer()
    if not isinstance(adapter, MEMOIRAdapter):
        raise RuntimeError("Failed to retrieve MEMOIR adapter after edit")
    setattr(model, "_memoir_adapter", adapter)

    return model, {}
