import hashlib
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import torch


def brackets_to_periods(name: str) -> str:
    return name.replace("[", ".").replace("]", "")


def parent_module(model, pname: str):
    components = pname.split(".")
    parent = model
    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component} in {pname}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]} in {pname}")
    return parent


def _sanitize_filename(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_")
    return "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-", "."))


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def get_context_templates(
    *,
    model,
    tok,
    length_params: Sequence[Sequence[int]],
    device: str,
    cache_dir: Path,
) -> List[str]:
    """
    Generate (or load) context templates used for MEMOIR augmentation.
    Writes the templates under `cache_dir` for reuse across runs.
    """
    model_id = getattr(getattr(model, "config", None), "_name_or_path", None) or "model"
    key = {
        "model": str(model_id),
        "length_params": [list(map(int, p)) for p in length_params],
    }
    key_bytes = json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    cache_name = _sanitize_filename(f"{model_id}_{hashlib.sha1(key_bytes).hexdigest()[:12]}")
    cache_path = cache_dir / f"{cache_name}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data

    prompt_tok = tok(
        ["I", "You", "Because", "Yes", "Q: "],
        padding=True,
        return_tensors="pt",
    ).to(device)

    templates: List[str] = []
    for length, n_gen in length_params:
        length = int(length)
        n_gen = int(n_gen)
        per_prompt = max(1, n_gen // 5)
        gen_token = model.generate(
            input_ids=prompt_tok["input_ids"],
            attention_mask=prompt_tok["attention_mask"],
            max_new_tokens=length,
            num_beams=per_prompt,
            num_return_sequences=per_prompt,
            pad_token_id=tok.eos_token_id,
        )
        templates += tok.batch_decode(gen_token, skip_special_tokens=True)

    templates = ["{}"] + [t + " {}" for t in templates]
    _atomic_write_json(cache_path, templates)
    return templates
