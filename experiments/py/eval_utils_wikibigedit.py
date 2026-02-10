"""
WikiBigEdit evaluation utilities.

We align with UltraEdit's recommended teacher-forcing token-level accuracy metrics:
- ES (edit success): update prompt -> ans
- GS (generalization): rephrase prompt -> ans
- LS (locality): loc prompt -> loc_ans

We reuse NAS's standard keys so checkpoint summaries keep working:
- rewrite_prompts_correct -> ES
- paraphrase_prompts_correct -> GS
- neighborhood_prompts_correct -> LS
"""

import typing
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_templates import fill_subject


@contextmanager
def _maybe_memoir_prompt_boundaries(model, boundaries: torch.Tensor):
    adapter = getattr(model, "_memoir_adapter", None)
    if adapter is None:
        yield
        return
    setattr(adapter, "last_prompt_token_loc_inference", boundaries)
    try:
        yield
    finally:
        if hasattr(adapter, "last_prompt_token_loc_inference"):
            delattr(adapter, "last_prompt_token_loc_inference")


def _token_level_accuracies(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: typing.List[str],
    answers: typing.List[str],
) -> typing.List[float]:
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must have the same length")
    if not prompts:
        return []

    device = next(model.parameters()).device

    normalized_answers = [
        a if a.startswith(" ") else (" " + a) for a in answers
    ]
    full_texts = [p + a for p, a in zip(prompts, normalized_answers)]

    prompt_lens = [len(ids) for ids in tok(prompts)["input_ids"]]
    full_tok = tok(full_texts, return_tensors="pt", padding=True).to(device)

    input_ids = full_tok["input_ids"][:, :-1]
    attention_mask = full_tok["attention_mask"][:, :-1]

    labels = full_tok["input_ids"][:, 1:].clone()
    labels[full_tok["attention_mask"][:, 1:] == 0] = -100
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, : max(prompt_len - 1, 0)] = -100

    boundaries = torch.tensor(
        [max(l - 1, 0) for l in prompt_lens],
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():
        with _maybe_memoir_prompt_boundaries(model, boundaries):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    preds = logits.argmax(-1)
    valid = labels != -100
    correct = (preds == labels) & valid
    denom = valid.sum(-1).to(torch.float32)
    num = correct.sum(-1).to(torch.float32)
    ratios = torch.where(denom > 0, num / denom, torch.zeros_like(num))
    return ratios.detach().cpu().numpy().tolist()


def compute_rewrite_quality_wikibigedit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips=None,
    vec=None,
) -> typing.Dict:
    req = record["requested_rewrite"]
    subject = req["subject"]

    rewrite_prompt = fill_subject(req["prompt"], subject)
    rewrite_answer = req["target_new"]["str"]

    paraphrase_prompts = record.get("paraphrase_prompts") or []
    if not paraphrase_prompts:
        raise ValueError(f"Missing paraphrase_prompts for case_id={record.get('case_id')}")
    paraphrase_prompt = paraphrase_prompts[0]

    neighborhood_prompts = record.get("neighborhood_prompts") or []
    if not neighborhood_prompts:
        raise ValueError(f"Missing neighborhood_prompts for case_id={record.get('case_id')}")
    neighborhood = neighborhood_prompts[0]
    loc_prompt = neighborhood["prompt"]
    loc_answer = neighborhood["target"]

    es, gs, ls = _token_level_accuracies(
        model,
        tok,
        [rewrite_prompt, paraphrase_prompt, loc_prompt],
        [rewrite_answer, rewrite_answer, loc_answer],
    )

    return {
        "rewrite_prompts_correct": [es],
        "paraphrase_prompts_correct": [gs],
        "neighborhood_prompts_correct": [ls],
        "ES": es,
        "GS": gs,
        "LS": ls,
    }
