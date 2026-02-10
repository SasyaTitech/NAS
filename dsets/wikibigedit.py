import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from torch.utils.data import Dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Dataset = object  # type: ignore[misc,assignment]

from prompt_templates import fill_subject


def _ensure_single_placeholder_template(prompt: str, subject: str) -> str:
    """
    Return a template containing exactly one supported subject placeholder.

    We prefer preserving an existing single "{}". If absent, we try to locate the subject
    span inside the prompt (exact or fuzzy). If we still cannot find it, we fall back to
    prepending the subject placeholder (so downstream code can still locate the subject
    token), instead of raising.
    """
    if "{}" in prompt:
        if prompt.count("{}") != 1:
            raise ValueError(
                f"WikiBigEdit prompt must contain exactly one '{{}}' placeholder; got {prompt.count('{}')} "
                f"for prompt={prompt!r}"
            )
        return prompt

    start = prompt.find(subject)
    if start < 0:
        tokens = [t for t in str(subject).split() if t]
        if len(tokens) >= 2:
            sep = r"(?:\s*[,;:()\[\]\"'“”‘’\-–—]?\s+)"
            pattern = r"\b" + re.escape(tokens[0]) + "".join(
                sep + re.escape(tok) for tok in tokens[1:]
            ) + r"\b"
            m = re.search(pattern, prompt, flags=re.IGNORECASE)
            if m is not None:
                start, end = m.span()
                templ = prompt[:start] + "{}" + prompt[end:]
                if templ.count("{}") != 1:
                    raise ValueError(
                        f"WikiBigEdit prompt template must contain exactly one '{{}}' placeholder; "
                        f"got {templ.count('{}')} for template={templ!r}"
                    )
                return templ

        # Last resort: ensure the subject appears once by construction.
        return "{} " + str(prompt)
    templ = prompt[:start] + "{}" + prompt[start + len(subject) :]
    if templ.count("{}") != 1:
        raise ValueError(
            f"WikiBigEdit prompt template must contain exactly one '{{}}' placeholder; got {templ.count('{}')} "
            f"for template={templ!r}"
        )
    return templ


class WikiBigEditDataset(Dataset):
    """
    NAS-format wrapper around the HuggingFace dataset `lukasthede/WikiBigEdit`.

    Default behavior mirrors UltraEdit's curated 17k splits by selecting a fixed list
    of indices from the HF `train` split (no dataset content is bundled in this repo).
    """

    def __init__(
        self,
        data_dir: str,
        tok=None,
        size: Optional[int] = None,
        *,
        subset: str = "full",
        hf_dataset: str = "lukasthede/WikiBigEdit",
        hf_split: str = "train",
        hf_revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **_: Any,
    ):
        # Import lazily so the rest of NAS works without the HF datasets dependency.
        try:
            from datasets import load_dataset  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "WikiBigEditDataset requires the HuggingFace `datasets` package. "
                "Install it (e.g. `pip install datasets`) or run inside the RunAE env."
            ) from e

        if subset != "full":
            raise ValueError(
                f"WikiBigEditDataset subset={subset!r} is not supported in this repo version; "
                "the default UltraEdit 17k subset logic was removed. Use subset='full'."
            )

        ds = load_dataset(hf_dataset, split=hf_split, revision=hf_revision, cache_dir=cache_dir)
        if size is not None:
            ds = ds.select(range(min(int(size), len(ds))))
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(*item.indices(len(self)))
            return [self[i] for i in indices]

        row = self._ds[int(item)]
        subject = row.get("subject")
        update = row.get("update")
        ans = row.get("ans")
        if not isinstance(subject, str) or not isinstance(update, str) or not isinstance(ans, str):
            raise ValueError(f"Invalid WikiBigEdit row at idx={item}: {row!r}")

        prompt_template = _ensure_single_placeholder_template(update, subject)

        rephrase = row.get("rephrase")
        loc = row.get("loc")
        loc_ans = row.get("loc_ans")
        mhop = row.get("mhop")
        mhop_ans = row.get("mhop_ans")
        personas = row.get("personas")
        tag = row.get("tag")

        if not isinstance(rephrase, str) or not rephrase:
            rephrase = fill_subject(prompt_template, subject)
        if not isinstance(loc, str) or not loc:
            loc = fill_subject(prompt_template, subject)
        if not isinstance(loc_ans, str) or not loc_ans:
            loc_ans = ans

        return {
            "case_id": int(item),
            "requested_rewrite": {
                "subject": subject,
                "prompt": prompt_template,
                "relation_id": row.get("relation_id"),
                "target_new": {"str": ans, "id": row.get("object_id")},
                "target_true": {"str": "", "id": None},
            },
            "paraphrase_prompts": [rephrase],
            "neighborhood_prompts": [{"prompt": loc, "target": loc_ans}],
            "wikibigedit": {
                "tag": tag,
                "personas": personas,
                "mhop": mhop,
                "mhop_ans": mhop_ans,
            },
        }
