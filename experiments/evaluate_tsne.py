# experiments/evaluate.py
# English comments only.

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.rome import ROMEHyperParams, apply_rome_to_model
from baselines.rome import repr_tools
from util import nethook
from util.globals import DATA_DIR, HPARAMS_DIR, KV_DIR, RESULTS_DIR

from dsets import CounterFactDataset, MENDQADataset, MQUAKEDataset

from prompt_templates import fill_subject

try:
    from dsets import MultiCounterFactDataset
except Exception:
    MultiCounterFactDataset = None


# ============================================================
# Base config (edit here; CLI overrides these)
# ============================================================

TSNE_MODE_DEFAULT = "factual"  # "factual" (AlphaEdit-aligned) | "wiki" (legacy)
TSNE_N_PROMPTS_DEFAULT = 1000
TSNE_PROMPT_BATCH_SIZE_DEFAULT = 16
TSNE_WIKI_NTOK_DEFAULT = 1000

SEED_DEFAULT = 1234


# =========================
# Determinism / seed
# =========================

def set_global_seed(seed: int, *, enforce_determinism: bool = True) -> None:
    """
    Set global seeds and common determinism knobs.

    Notes:
      - We avoid torch.use_deterministic_algorithms(True) because some kernels may raise.
      - We enforce practical determinism by:
          * disabling TF32
          * configuring SDPA to math backend (no flash / no mem-efficient) if available
      - If enforce_determinism=True and SDPA controls are missing, we raise.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0")

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if enforce_determinism:
        if not hasattr(torch.backends, "cuda"):
            raise RuntimeError("torch.backends.cuda is missing; cannot configure SDPA determinism.")
        cuda_be = torch.backends.cuda
        needed = ["enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"]
        for name in needed:
            if not hasattr(cuda_be, name):
                raise RuntimeError(f"Missing torch.backends.cuda.{name}; cannot enforce deterministic SDPA backend.")
        cuda_be.enable_flash_sdp(False)
        cuda_be.enable_mem_efficient_sdp(False)
        cuda_be.enable_math_sdp(True)


# =========================
# Dataset registry (editing only)
# =========================

def _get_dataset_class(ds_name: str):
    """
    Resolve dataset class strictly.

    Notes:
      - "mcf" requires MultiCounterFactDataset; if missing, we throw.
    """
    if ds_name == "mcf":
        if MultiCounterFactDataset is None:
            raise RuntimeError("ds_name=mcf requested but dsets.MultiCounterFactDataset is not available.")
        return MultiCounterFactDataset

    if ds_name == "cf":
        return CounterFactDataset
    if ds_name == "zsre":
        return MENDQADataset
    if ds_name == "mquake":
        return MQUAKEDataset

    raise ValueError(f"Unknown ds_name: {ds_name}")


# =========================
# Helpers
# =========================

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
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _normalize_case_id(case_id):
    if isinstance(case_id, np.generic):
        return int(case_id)
    if case_id is None:
        return None
    try:
        return int(case_id)
    except Exception:
        return case_id


def _apply_hparam_overrides(hparams: ROMEHyperParams, overrides: List[str]):
    if not overrides:
        return {}
    before = asdict(hparams)
    changed: List[str] = []

    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Invalid --hparam {spec!r}; expected key=value")
        key, raw = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --hparam {spec!r}; empty key")
        if key not in before:
            raise ValueError(f"Unknown hyperparameter {key!r} for {type(hparams).__name__}")
        setattr(hparams, key, _parse_override_value(raw))
        changed.append(key)

    after = asdict(hparams)
    return {k: {"before": before.get(k), "after": after.get(k)} for k in changed}


def _get_num_hidden_layers(model) -> int:
    """Best-effort extraction for common HF configs."""
    cfg = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if cfg is not None and hasattr(cfg, attr):
            n = int(getattr(cfg, attr))
            if n > 0:
                return n
    raise RuntimeError("Cannot determine num hidden layers from model.config.")


# =========================
# AlphaEdit-aligned: factual prompt bank (1000 prompts)
# =========================

def _render_factual_prompt_from_rr(rr: dict) -> str:
    """
    Build a concrete factual prompt string from a requested_rewrite dict.

    We try common placeholder styles:
      - "{}"
      - "{subject}"
      - "<subject>"
    """
    prompt = str(rr.get("prompt") or "")
    subject = rr.get("subject", None)
    if subject is None:
        return prompt

    if any(ph in prompt for ph in ("{}", "{subject}", "{0}")):
        return fill_subject(prompt, str(subject))
    if "<subject>" in prompt:
        return prompt.replace("<subject>", str(subject))
    return prompt


def _canonicalize_subject_prompt_template(rr_prompt: str) -> str:
    """
    Convert common placeholder styles into a canonical template with exactly one "{}".

    Supported input formats:
      - "{} ...": already canonical
      - "{subject} ...": converted to "{} ..."
      - "<subject> ...": converted to "{} ..."
    """
    prompt = str(rr_prompt or "")
    if "{}" in prompt:
        if prompt.count("{}") != 1:
            raise RuntimeError(f"Prompt template must contain exactly one '{{}}': {prompt!r}")
        return prompt

    if "{subject}" in prompt:
        if prompt.count("{subject}") != 1:
            raise RuntimeError(f"Prompt template must contain exactly one '{{subject}}': {prompt!r}")
        tmp = prompt.replace("{subject}", "{}")
        if tmp.count("{}") != 1:
            raise RuntimeError(f"Prompt template canonicalization failed: {prompt!r}")
        return tmp

    if "<subject>" in prompt:
        if prompt.count("<subject>") != 1:
            raise RuntimeError(f"Prompt template must contain exactly one '<subject>': {prompt!r}")
        tmp = prompt.replace("<subject>", "{}")
        if tmp.count("{}") != 1:
            raise RuntimeError(f"Prompt template canonicalization failed: {prompt!r}")
        return tmp

    raise RuntimeError(
        "Prompt template has no supported placeholder. "
        "Expected one of: '{}', '{subject}', '<subject>'. "
        f"Got: {prompt!r}"
    )


def _get_first_requested_rewrite(record: dict) -> dict:
    rr = record.get("requested_rewrite", None)
    if rr is None:
        raise RuntimeError("Record missing requested_rewrite.")
    if isinstance(rr, list):
        if not rr:
            raise RuntimeError("requested_rewrite is an empty list.")
        return rr[0]
    if isinstance(rr, dict):
        return rr
    raise RuntimeError(f"Unexpected requested_rewrite type: {type(rr)}")


def _get_all_requested_rewrites(record: dict) -> List[dict]:
    rr = record.get("requested_rewrite", None)
    if rr is None:
        raise RuntimeError("Record missing requested_rewrite.")
    if isinstance(rr, list):
        if not rr:
            raise RuntimeError("requested_rewrite is an empty list.")
        return rr
    if isinstance(rr, dict):
        return [rr]
    raise RuntimeError(f"Unexpected requested_rewrite type: {type(rr)}")


def _load_or_build_factual_prompt_cache(
        *,
        ds,
        ds_name: str,
        dataset_size_limit: Optional[int],
        n_prompts: int,
        seed: int,
        cache_path: Path,
        exclude_first_n: int,
        exclude_subjects: Optional[Set[str]] = None,
) -> dict:
    """
    Cache schema (json):
      {
        "version": 3,
        "ds_name": "...",
        "dataset_size_limit": 1000 or null,
        "seed": 1234,
        "n_prompts": 1000,
        "exclude_first_n": 1000,
        "sampling": "reverse_scan_unique_subject",
        "unique_subject": true,
        "exclude_edited_subjects": true,
        "indices": [...],
        "case_ids": [...],
        "subjects": ["...", ...],
        "prompt_templates": ["{} ...", ...],
        "prompts": ["rendered prompt...", ...]
      }
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if int(obj.get("version", 0)) == 3:
            if str(obj.get("ds_name")) != str(ds_name):
                raise RuntimeError(f"Prompt cache ds_name mismatch: {obj.get('ds_name')} vs {ds_name}")
            if obj.get("dataset_size_limit", None) != dataset_size_limit:
                raise RuntimeError(
                    f"Prompt cache dataset_size_limit mismatch: {obj.get('dataset_size_limit')} vs {dataset_size_limit}"
                )
            if int(obj.get("seed")) != int(seed):
                raise RuntimeError(f"Prompt cache seed mismatch: {obj.get('seed')} vs {seed}")
            if int(obj.get("n_prompts")) != int(n_prompts):
                raise RuntimeError(f"Prompt cache n_prompts mismatch: {obj.get('n_prompts')} vs {n_prompts}")
            if int(obj.get("exclude_first_n", -1)) != int(exclude_first_n):
                raise RuntimeError(
                    f"Prompt cache exclude_first_n mismatch: {obj.get('exclude_first_n')} vs {exclude_first_n}"
                )

            prompts = obj.get("prompts", None)
            templates = obj.get("prompt_templates", None)
            subjects = obj.get("subjects", None)
            indices = obj.get("indices", None)
            if not isinstance(prompts, list) or not isinstance(indices, list) or not isinstance(templates, list) or not isinstance(subjects, list):
                raise RuntimeError(f"Prompt cache corrupted: {cache_path}")
            if len(prompts) != n_prompts or len(indices) != n_prompts or len(templates) != n_prompts or len(subjects) != n_prompts:
                raise RuntimeError(f"Prompt cache length mismatch: {cache_path}")
            return obj

        print(f"[tsne_v] Prompt cache {cache_path} has unsupported version={obj.get('version')}; rebuilding.")

    if not hasattr(ds, "__len__"):
        raise RuntimeError("Dataset has no __len__; cannot sample prompts deterministically.")
    if not hasattr(ds, "__getitem__"):
        raise RuntimeError("Dataset has no __getitem__; cannot index by sampled indices.")

    n_total = int(len(ds))
    if n_total < n_prompts:
        raise RuntimeError(f"Dataset too small: len(ds)={n_total} < n_prompts={n_prompts}")

    exclude_subjects = exclude_subjects or set()
    if exclude_first_n < 0:
        raise ValueError("exclude_first_n must be >= 0")
    if exclude_first_n > n_total:
        raise ValueError(f"exclude_first_n={exclude_first_n} > len(ds)={n_total}")

    indices: List[int] = []
    prompts: List[str] = []
    prompt_templates: List[str] = []
    subjects: List[str] = []
    case_ids: List[Optional[Union[int, str]]] = []

    # Deterministic reverse scan:
    # - Avoid edited region (first exclude_first_n records).
    # - Avoid any subject that appears in the edited region.
    # - Enforce unique subjects among sampled prompts.
    seen_subjects: Set[str] = set()
    skipped_bad_template = 0
    for i in range(n_total - 1, exclude_first_n - 1, -1):
        rec = ds[int(i)]
        rr0 = _get_first_requested_rewrite(rec)
        subject = rr0.get("subject", None)
        if not isinstance(subject, str) or not subject:
            continue
        if subject in exclude_subjects:
            continue
        if subject in seen_subjects:
            continue

        raw_prompt = rr0.get("prompt", "")
        try:
            tmpl = _canonicalize_subject_prompt_template(str(raw_prompt))
            rendered = fill_subject(tmpl, subject)
        except Exception:
            skipped_bad_template += 1
            continue

        indices.append(int(i))
        prompts.append(rendered)
        prompt_templates.append(tmpl)
        subjects.append(subject)
        case_ids.append(_normalize_case_id(rec.get("case_id")))
        seen_subjects.add(subject)
        if len(prompts) >= int(n_prompts):
            break

    if len(prompts) != int(n_prompts):
        raise RuntimeError(
            "Not enough eligible prompts after filtering. "
            f"Need n_prompts={n_prompts}, got {len(prompts)}. "
            f"exclude_first_n={exclude_first_n} exclude_subjects={len(exclude_subjects)} "
            f"skipped_bad_template={skipped_bad_template}"
        )

    obj = {
        "version": 3,
        "ds_name": str(ds_name),
        "dataset_size_limit": dataset_size_limit,
        "seed": int(seed),
        "n_prompts": int(n_prompts),
        "exclude_first_n": int(exclude_first_n),
        "sampling": "reverse_scan_unique_subject",
        "unique_subject": True,
        "exclude_edited_subjects": True,
        "indices": indices,
        "case_ids": case_ids,
        "subjects": subjects,
        "prompt_templates": prompt_templates,
        "prompts": prompts,
    }
    _atomic_write_json(cache_path, obj)
    return obj


def _tokenize_prompt_bank(
        *,
        tok: AutoTokenizer,
        prompt_cache: dict,
) -> Dict[str, torch.Tensor]:
    """
    Returns a CPU tensor bank:
      - input_ids: LongTensor [N, L]
      - attention_mask: LongTensor [N, L]
    """
    prompts = prompt_cache.get("prompts", None)
    prompt_templates = prompt_cache.get("prompt_templates", None)
    subjects = prompt_cache.get("subjects", None)
    if not isinstance(prompts, list) or not isinstance(prompt_templates, list) or not isinstance(subjects, list):
        raise RuntimeError("Prompt cache missing prompts/prompt_templates/subjects (expected version=3 cache).")
    if len(prompts) != len(prompt_templates) or len(prompts) != len(subjects):
        raise RuntimeError("Prompt cache length mismatch (prompts/templates/subjects).")

    # Re-render from templates for consistency with subject index computation.
    prompts = [fill_subject(str(prompt_templates[i]), str(subjects[i])) for i in range(len(subjects))]
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(torch.long).contiguous()
    attn = enc.get("attention_mask", None)
    if attn is None:
        raise RuntimeError("Tokenizer did not return attention_mask.")
    attention_mask = attn.to(torch.long).contiguous()

    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise RuntimeError(f"Unexpected token bank shapes: {tuple(input_ids.shape)}, {tuple(attention_mask.shape)}")
    if input_ids.shape != attention_mask.shape:
        raise RuntimeError("input_ids and attention_mask shape mismatch.")

    idxs = repr_tools.get_words_idxs_in_templates(
        tok=tok,
        context_templates=[str(t) for t in prompt_templates],
        words=[str(s) for s in subjects],
        subtoken="last",
    )
    if not isinstance(idxs, list) or len(idxs) != len(prompts):
        raise RuntimeError("Unexpected subject index output shape from repr_tools.")
    flat: List[int] = []
    for item in idxs:
        if not isinstance(item, list) or len(item) != 1:
            raise RuntimeError(f"Unexpected subject idx entry: {item}")
        flat.append(int(item[0]))

    subject_idxs = torch.tensor(flat, dtype=torch.long).contiguous()
    if subject_idxs.ndim != 1 or subject_idxs.numel() != input_ids.size(0):
        raise RuntimeError("subject_idxs shape mismatch.")
    if (subject_idxs < 0).any() or (subject_idxs >= input_ids.size(1)).any():
        raise RuntimeError("subject_idxs out of bounds for tokenized prompts (possibly due to truncation).")

    return {"input_ids": input_ids, "attention_mask": attention_mask, "subject_idxs": subject_idxs}


# =========================
# Legacy: Wikipedia workload (optional)
# =========================

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_TITLE_FIXED = "Artificial_intelligence"


def _fetch_wikipedia_extract(title: str, timeout_sec: int = 30) -> str:
    """Fetch plaintext extract via MediaWiki API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "format": "json",
        "redirects": "1",
        "titles": title,
    }
    url = _WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "rome-tsne-v-collector/1.0 (research)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as r:
        data = r.read()
    payload = json.loads(data.decode("utf-8", errors="replace"))
    pages = payload.get("query", {}).get("pages", {})
    if not isinstance(pages, dict) or not pages:
        raise RuntimeError("Wikipedia API returned no pages.")
    page = next(iter(pages.values()))
    text = (page.get("extract") or "").strip()
    if not text:
        raise RuntimeError(f"Wikipedia extract empty for title={title!r}.")
    return text


def _load_or_fetch_wiki_text(*, cache_path: Path) -> Tuple[str, str]:
    """
    Load cached Wikipedia text if present, otherwise fetch and cache a fixed page.

    Cache format:
      {"title": "...", "text": "..."}
    """
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        title = str(obj.get("title") or "")
        text = str(obj.get("text") or "")
        if title and text:
            return title, text
        raise RuntimeError(f"Invalid cache file: {cache_path}")

    title = _WIKI_TITLE_FIXED
    text = _fetch_wikipedia_extract(title)
    _atomic_write_json(cache_path, {"title": title, "text": text})
    return title, text


def _prepare_wikipedia_token_ids(
        *,
        run_dir: Path,
        tok: AutoTokenizer,
        n_tokens: int = 1000,
        wiki_cache_path: Optional[Path] = None,
) -> Tuple[str, torch.LongTensor]:
    """
    Returns (title, input_ids) where input_ids is shape [1, n_tokens+1].
    """
    tsne_dir = run_dir / "tsne_v"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    cache_path = wiki_cache_path if wiki_cache_path is not None else (Path(DATA_DIR) / "tsne_wikipedia_source.json")
    title, text = _load_or_fetch_wiki_text(cache_path=cache_path)

    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise RuntimeError(f"Unexpected tokenization shape: {tuple(input_ids.shape)}")

    need = n_tokens + 1
    if input_ids.size(1) < need:
        raise RuntimeError(
            f"Wikipedia text too short: got {input_ids.size(1)} tokens, need {need}. Cache file: {cache_path}"
        )

    input_ids = input_ids[:, :need].contiguous()
    return title, input_ids


# =========================
# t-SNE v collector
# =========================

def _collect_tsne_v(
        *,
        model,
        tok: AutoTokenizer,
        hparams: ROMEHyperParams,
        run_dir: Path,
        edit_step: int,
        seed: int,
        mode: str,
        # factual mode:
        prompt_bank: Optional[Dict[str, torch.Tensor]] = None,
        prompt_cache_path: Optional[Path] = None,
        n_prompts: int = 1000,
        batch_size: int = 16,
        # wiki mode:
        n_tokens: int = 1000,
        wiki_cache_path: Optional[Path] = None,
) -> Path:
    """
    Collect v vectors for t-SNE.

    mode:
      - "factual": sample dimension = 1000 prompts (AlphaEdit-aligned)
      - "wiki":    sample dimension = 1000 token-time steps (legacy)

    edit_step convention:
      - edit_step == 0: base (pure) model before any edits (step_0000.pt)
      - edit_step >= 1: model after the edit_step-th sequential edit (1-indexed)
    """
    if edit_step < 0:
        raise ValueError("edit_step must be >= 0")

    tsne_dir = run_dir / "tsne_v"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    out_path = tsne_dir / f"step_{int(edit_step):04d}.pt"

    edit_layers = list(getattr(hparams, "layers", []))
    if not edit_layers:
        raise RuntimeError("ROMEHyperParams.layers is empty; cannot decide which layers to hook.")

    module_names: List[str] = [hparams.rewrite_module_tmp.format(int(layer)) for layer in edit_layers]

    last_layer = _get_num_hidden_layers(model) - 1

    model.eval()
    t0 = time()

    if mode == "factual":
        if prompt_bank is None:
            raise RuntimeError("mode=factual requires prompt_bank (tokenized).")
        input_ids_all = prompt_bank["input_ids"]
        attn_all = prompt_bank["attention_mask"]
        if input_ids_all.size(0) < n_prompts:
            raise RuntimeError(f"prompt_bank has N={input_ids_all.size(0)} < n_prompts={n_prompts}")

        input_ids_all = input_ids_all[:n_prompts, :]
        attn_all = attn_all[:n_prompts, :]
        subj_idx_all = prompt_bank.get("subject_idxs", None)
        if not isinstance(subj_idx_all, torch.Tensor):
            raise RuntimeError("mode=factual requires prompt_bank['subject_idxs'] (subject_last indices).")
        subj_idx_all = subj_idx_all[:n_prompts].to(torch.long).contiguous()

        v_by_module: Dict[str, List[torch.Tensor]] = {m: [] for m in module_names}

        with torch.inference_mode():
            with nethook.TraceDict(
                    module=model,
                    layers=module_names,
                    retain_output=True,
                    retain_input=False,
                    clone=False,
                    detach=True,
                    retain_grad=False,
                    edit_output=None,
                    stop=False,
            ) as traces:
                for start in range(0, n_prompts, batch_size):
                    end = min(n_prompts, start + batch_size)
                    ids = input_ids_all[start:end].to(model.device)
                    am = attn_all[start:end].to(model.device)
                    subj_idx = subj_idx_all[start:end].to(model.device)

                    _ = model(input_ids=ids, attention_mask=am, use_cache=False)

                    bsz = ids.size(0)
                    row = torch.arange(bsz, device=model.device)

                    for m in module_names:
                        v = traces[m].output
                        if isinstance(v, (tuple, list)):
                            v = v[0]
                        if not isinstance(v, torch.Tensor):
                            raise RuntimeError(f"Hooked output for {m!r} is not a Tensor: {type(v)}")
                        if v.ndim != 3:
                            raise RuntimeError(f"Expected hooked output [B, L, D] for {m!r}, got {tuple(v.shape)}")

                        if (subj_idx < 0).any() or (subj_idx >= v.size(1)).any():
                            raise RuntimeError("subject_idxs out of bounds for hooked sequence length.")
                        vt = v[row, subj_idx, :]  # [B, D]
                        for j in range(vt.size(0)):
                            v_by_module[m].append(vt[j].detach().to("cpu", dtype=torch.float16))

        v_stacked: Dict[str, torch.Tensor] = {m: torch.stack(seq, dim=0) for m, seq in v_by_module.items()}

        payload = {
            "edit_step": int(edit_step),
            "is_base_model": (int(edit_step) == 0),
            "step": int(edit_step),

            "seed": int(seed),
            "source": {
                "kind": "factual_prompts",
                "n_prompts": int(n_prompts),
                "batch_size": int(batch_size),
                "prompt_cache_path": (str(prompt_cache_path) if prompt_cache_path is not None else ""),
                "token_strategy": "subject_last",
            },

            "modules": module_names,
            "edited_layers": [int(x) for x in edit_layers],
            "last_layer": int(last_layer),

            "v": v_stacked,
            "elapsed_sec": float(time() - t0),
        }
        torch.save(payload, out_path)
        return out_path

    if mode == "wiki":
        title, wiki_ids = _prepare_wikipedia_token_ids(
            run_dir=run_dir,
            tok=tok,
            n_tokens=n_tokens,
            wiki_cache_path=wiki_cache_path,
        )
        wiki_ids = wiki_ids.to(model.device)

        v_by_module: Dict[str, List[torch.Tensor]] = {m: [] for m in module_names}
        pred_ids: List[int] = []

        with torch.inference_mode():
            past = None
            with nethook.TraceDict(
                    module=model,
                    layers=module_names,
                    retain_output=True,
                    retain_input=False,
                    clone=False,
                    detach=True,
                    retain_grad=False,
                    edit_output=None,
                    stop=False,
            ) as traces:
                for t in range(n_tokens):
                    cur = wiki_ids[:, t:t + 1]
                    out = model(input_ids=cur, use_cache=True, past_key_values=past)
                    past = out.past_key_values

                    logits = out.logits[:, -1, :]
                    pred = int(torch.argmax(logits, dim=-1).item())
                    pred_ids.append(pred)

                    for m in module_names:
                        v = traces[m].output
                        if isinstance(v, (tuple, list)):
                            v = v[0]
                        if not isinstance(v, torch.Tensor):
                            raise RuntimeError(f"Hooked output for {m!r} is not a Tensor: {type(v)}")

                        if v.ndim == 3:
                            vt = v[0, -1]
                        elif v.ndim == 2:
                            vt = v[0]
                        else:
                            raise RuntimeError(f"Unexpected hooked tensor shape for {m!r}: {tuple(v.shape)}")

                        v_by_module[m].append(vt.detach().to("cpu", dtype=torch.float16))

        v_stacked: Dict[str, torch.Tensor] = {m: torch.stack(seq, dim=0) for m, seq in v_by_module.items()}

        payload = {
            "edit_step": int(edit_step),
            "is_base_model": (int(edit_step) == 0),
            "step": int(edit_step),

            "seed": int(seed),
            "source": {"kind": "wikipedia_tokens", "title": title, "n_tokens": int(n_tokens)},

            "modules": module_names,
            "edited_layers": [int(x) for x in edit_layers],
            "last_layer": int(last_layer),

            "input_ids": wiki_ids[0, : n_tokens + 1].detach().to("cpu"),
            "target_ids": wiki_ids[0, 1: n_tokens + 1].detach().to("cpu"),
            "pred_ids": torch.tensor(pred_ids, dtype=torch.long),

            "v": v_stacked,
            "elapsed_sec": float(time() - t0),
        }
        torch.save(payload, out_path)
        return out_path

    raise ValueError(f"Unknown tsne mode: {mode!r}")


# =========================
# Main runner (ROME only, sequential edits only)
# =========================

def main(
        *,
        model_name: str,
        hparams_fname: str,
        ds_name: str,
        dataset_size_limit: Optional[int],
        dir_name: str,
        use_nas: bool,
        use_cache: bool = False,
        edit_log: bool = False,
        hparam_overrides: Optional[List[str]] = None,
        eval_step: int = 0,
        seed: int = SEED_DEFAULT,
        wiki_cache_path: Optional[str] = None,
        tsne_mode: str = TSNE_MODE_DEFAULT,
        tsne_n_prompts: int = TSNE_N_PROMPTS_DEFAULT,
        tsne_batch_size: int = TSNE_PROMPT_BATCH_SIZE_DEFAULT,
        tsne_prompt_cache_path: str = "",
        tsne_wiki_n_tokens: int = TSNE_WIKI_NTOK_DEFAULT,
):
    """
    Sequential editing runner for ROME only.

    Hard constraints:
      - num_edits is always 1 (sequential editing mainline).
      - No silent error handling: all errors are raised.

    t-SNE dump:
      - step_0000.pt is ALWAYS the base (pure) model (no edits).
      - step_0001.pt is after the 1st edit, etc.
    """
    set_global_seed(seed, enforce_determinism=True)

    if tsne_mode not in {"factual", "wiki"}:
        raise ValueError(f"Invalid tsne_mode: {tsne_mode}")

    ds_class = _get_dataset_class(ds_name)

    # Create a new run dir always (strict).
    alg_dir = Path(RESULTS_DIR) / dir_name
    if alg_dir.exists():
        id_list = [
            int(str(x).split("_")[-1])
            for x in alg_dir.iterdir()
            if str(x).split("_")[-1].isnumeric()
        ]
        run_id = 0 if not id_list else max(id_list) + 1
    else:
        run_id = 0

    run_dir = Path(RESULTS_DIR) / dir_name / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Load hyperparameters.
    params_path = Path(HPARAMS_DIR) / "ROME" / hparams_fname
    hparams = ROMEHyperParams.from_json(params_path)
    hparam_overrides = hparam_overrides or []

    if hparam_overrides:
        changes = _apply_hparam_overrides(hparams, hparam_overrides)
        shutil.copyfile(params_path, run_dir / "params_base.json")
        with open(run_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(asdict(hparams)), f, indent=2, ensure_ascii=False)
        _atomic_write_json(
            run_dir / "hparam_overrides.json",
            {"base_params": str(params_path), "overrides": hparam_overrides, "changes": changes},
            )
        print(f"Applied hyperparameter overrides: {changes}")
    else:
        shutil.copyfile(params_path, run_dir / "params.json")

    # Override use_nas (strict).
    if not hasattr(hparams, "use_nas"):
        raise RuntimeError("ROMEHyperParams has no field 'use_nas', but --use_nas override was requested.")
    print(f"Overriding use_nas in hparams: {getattr(hparams, 'use_nas')} -> {use_nas}")
    hparams.use_nas = bool(use_nas)

    print(f"Executing ROME with parameters {hparams}")

    # Model / tokenizer.
    print("Instantiating model")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    # Dataset for sequential editing (prefix). We also keep a pool dataset for t-SNE prompt sampling.
    ds_edit = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    ds_pool = ds_edit
    try:
        ds_pool = ds_class(DATA_DIR, tok=tok, size=None)
    except Exception:
        ds_pool = ds_edit

    exclude_first_n = int(len(ds_edit)) if dataset_size_limit is not None else 0
    edited_subjects: Set[str] = set()
    if exclude_first_n > 0:
        if not hasattr(ds_edit, "__len__") or not hasattr(ds_edit, "__getitem__"):
            raise RuntimeError("Dataset has no __len__/__getitem__; cannot build edited_subjects.")
        for i in range(int(len(ds_edit))):
            rec = ds_edit[int(i)]
            for rr in _get_all_requested_rewrites(rec):
                subject = rr.get("subject", None)
                if isinstance(subject, str) and subject:
                    edited_subjects.add(subject)
        print(f"[tsne_v] Excluding subjects from edited prefix: {len(edited_subjects)} unique subjects")

    # Cache template for ROME (optional).
    cache_template = None
    if use_cache:
        cache_template = (
                Path(KV_DIR)
                / f"{model_name.replace('/', '_')}_ROME"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Edit logs (optional).
    edit_log_kwargs = {}
    if edit_log:
        edit_log_dir = run_dir / "edit_logs"
        edit_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Edit logs directory: {edit_log_dir}")
        edit_log_kwargs = {"edit_log_dir": str(edit_log_dir)}

    # Parse wiki cache path.
    wiki_cache = None
    if wiki_cache_path is not None and str(wiki_cache_path).strip():
        wiki_cache = Path(wiki_cache_path).expanduser().resolve()
    else:
        wiki_cache = None

    # Prepare t-SNE factual prompt bank (AlphaEdit-aligned) once per run.
    prompt_cache_path = None
    prompt_cache = None
    prompt_bank = None
    if int(eval_step) > 0 and tsne_mode == "factual":
        if tsne_n_prompts <= 0:
            raise ValueError("--tsne_n_prompts must be > 0 for factual mode.")
        if tsne_batch_size <= 0:
            raise ValueError("--tsne_batch_size must be > 0 for factual mode.")

        if str(tsne_prompt_cache_path).strip():
            prompt_cache_path = Path(tsne_prompt_cache_path).expanduser().resolve()
        else:
            lim_tag = "all" if dataset_size_limit is None else str(int(dataset_size_limit))
            prompt_cache_path = Path(DATA_DIR) / f"tsne_factual_prompts_{ds_name}_lim{lim_tag}_seed{seed}_n{tsne_n_prompts}.json"

        prompt_cache = _load_or_build_factual_prompt_cache(
            ds=ds_pool,
            ds_name=ds_name,
            dataset_size_limit=dataset_size_limit,
            n_prompts=tsne_n_prompts,
            seed=seed,
            cache_path=prompt_cache_path,
            exclude_first_n=exclude_first_n,
            exclude_subjects=edited_subjects,
        )
        prompt_bank = _tokenize_prompt_bank(tok=tok, prompt_cache=prompt_cache)
        _atomic_write_json(run_dir / "tsne_v" / "prompt_cache_ref.json", {"path": str(prompt_cache_path)})
        print(f"[tsne_v] Using factual prompt cache: {prompt_cache_path}")

    # Base dump (step_0000) if enabled.
    if int(eval_step) > 0:
        out_path0 = _collect_tsne_v(
            model=model,
            tok=tok,
            hparams=hparams,
            run_dir=run_dir,
            edit_step=0,
            seed=seed,
            mode=tsne_mode,
            prompt_bank=prompt_bank,
            prompt_cache_path=prompt_cache_path,
            n_prompts=tsne_n_prompts,
            batch_size=tsne_batch_size,
            n_tokens=tsne_wiki_n_tokens,
            wiki_cache_path=wiki_cache,
        )
        print(f"[tsne_v] Saved BASE (pure) dump at edit_step=0: {out_path0}")

    # Sequential editing loop: edit_step is 1-indexed.
    edit_step_count = 0
    for record in ds_edit:
        edit_step_count += 1
        case_id = record.get("case_id")
        print("=" * 66)
        print(f"[edit] edit_step={edit_step_count} case_id={case_id}")
        print("=" * 66)

        rr = record["requested_rewrite"]
        rr_list = rr if isinstance(rr, list) else [rr]
        reqs = [{"case_id": case_id, **rewrite_dict} for rewrite_dict in rr_list]

        etc_args = dict(cache_template=cache_template) if cache_template is not None else dict()

        start = time()
        edited_model, _ = apply_rome_to_model(
            model,
            tok,
            reqs,
            hparams,
            return_orig_weights=False,
            **etc_args,
            **edit_log_kwargs,
        )
        model = edited_model
        exec_time = time() - start
        print(f"[edit] Execution took {exec_time:.3f}s")

        # Periodic v-dump (strict).
        if int(eval_step) > 0 and (edit_step_count % int(eval_step) == 0):
            out_path = _collect_tsne_v(
                model=model,
                tok=tok,
                hparams=hparams,
                run_dir=run_dir,
                edit_step=edit_step_count,
                seed=seed,
                mode=tsne_mode,
                prompt_bank=prompt_bank,
                prompt_cache_path=prompt_cache_path,
                n_prompts=tsne_n_prompts,
                batch_size=tsne_batch_size,
                n_tokens=tsne_wiki_n_tokens,
                wiki_cache_path=wiki_cache,
            )
            print(f"[tsne_v] Saved dump at edit_step={edit_step_count}: {out_path}")

    # Final dump if enabled and not already dumped at the last periodic step.
    if int(eval_step) > 0 and (edit_step_count % int(eval_step) != 0):
        out_path = _collect_tsne_v(
            model=model,
            tok=tok,
            hparams=hparams,
            run_dir=run_dir,
            edit_step=edit_step_count,
            seed=seed,
            mode=tsne_mode,
            prompt_bank=prompt_bank,
            prompt_cache_path=prompt_cache_path,
            n_prompts=tsne_n_prompts,
            batch_size=tsne_batch_size,
            n_tokens=tsne_wiki_n_tokens,
            wiki_cache_path=wiki_cache,
        )
        print(f"[tsne_v] Saved final dump at edit_step={edit_step_count}: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ROME only.
    parser.add_argument("--alg_name", choices=["ROME"], default="ROME", required=True)

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--hparams_fname", type=str, required=True)

    parser.add_argument("--hparam", action="append", default=[])

    parser.add_argument("--ds_name", choices=["cf", "mcf", "zsre", "mquake"], default="cf")
    parser.add_argument("--dataset_size_limit", type=int, default=None)

    # Keep for CLI compatibility, but enforce sequential mainline.
    parser.add_argument("--num_edits", type=int, default=1)

    parser.add_argument("--use_cache", dest="use_cache", action="store_true")
    parser.add_argument("--edit_log", dest="edit_log", action="store_true")

    parser.add_argument(
        "--eval_step",
        type=int,
        default=0,
        help="Every N edit steps, dump tsne_v. 0 disables.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help="Global random seed used for reproducible step_0000 dumps and prompt sampling.",
    )

    # t-SNE alignment with AlphaEdit
    parser.add_argument("--tsne_mode", choices=["factual", "wiki"], default=TSNE_MODE_DEFAULT)
    parser.add_argument("--tsne_n_prompts", type=int, default=TSNE_N_PROMPTS_DEFAULT)
    parser.add_argument("--tsne_batch_size", type=int, default=TSNE_PROMPT_BATCH_SIZE_DEFAULT)
    parser.add_argument(
        "--tsne_prompt_cache_path",
        type=str,
        default="",
        help="Optional shared prompt cache JSON path. Empty => default under DATA_DIR.",
    )

    # Legacy wiki mode controls (only used if --tsne_mode=wiki)
    parser.add_argument(
        "--wiki_cache_path",
        type=str,
        default="",
        help="Path to Wikipedia cache JSON. Empty => use DATA_DIR/tsne_wikipedia_source.json.",
    )
    parser.add_argument("--tsne_wiki_n_tokens", type=int, default=TSNE_WIKI_NTOK_DEFAULT)

    parser.add_argument(
        "--use_nas",
        action="store_true",
        help="Whether to use NAS. Overrides the value in the hparams file.",
    )

    args = parser.parse_args()

    if int(args.num_edits) != 1:
        raise ValueError("Sequential editing mainline: --num_edits must be 1.")

    main(
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        dir_name="ROME",
        use_nas=bool(args.use_nas),
        use_cache=args.use_cache,
        edit_log=args.edit_log,
        hparam_overrides=args.hparam,
        eval_step=args.eval_step,
        seed=args.seed,
        wiki_cache_path=args.wiki_cache_path,
        tsne_mode=args.tsne_mode,
        tsne_n_prompts=args.tsne_n_prompts,
        tsne_batch_size=args.tsne_batch_size,
        tsne_prompt_cache_path=args.tsne_prompt_cache_path,
        tsne_wiki_n_tokens=args.tsne_wiki_n_tokens,
    )

# Example (AlphaEdit-aligned factual prompts):
# CUDA_VISIBLE_DEVICES=0 python3 -m experiments.evaluate \
#   --alg_name=ROME \
#   --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
#   --hparams_fname=Llama3-8B.json \
#   --ds_name=mcf \
#   --dataset_size_limit=1000 \
#   --eval_step=20 \
#   --seed=1234 \
#   --tsne_mode=factual \
#   --tsne_n_prompts=1000 \
#   --tsne_batch_size=16
#
# Vanilla vs NAS runs: run twice, second one add --use_nas, but keep same seed/ds/limit and prompt_cache_path if desired.
