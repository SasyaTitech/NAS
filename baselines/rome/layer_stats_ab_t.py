import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import (
    CombinedStat,
    Mean,
    NormMean,
    SecondMoment,
    SecondMoment_AB_T,
    tally,
)
from util import nethook

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

_WIKITEXT_CONFIG = "wikitext-103-raw-v1"
_WIKIPEDIA_CONFIG_CANDIDATES = ("20220301.en", "20200501.en")


def _load_hf_text_dataset(ds_name: str):
    if ds_name == "wikitext":
        return load_dataset(ds_name, _WIKITEXT_CONFIG)
    if ds_name == "wikipedia":
        last_err = None
        for cfg in _WIKIPEDIA_CONFIG_CANDIDATES:
            try:
                return load_dataset(ds_name, cfg)
            except ValueError as e:
                last_err = e
        if last_err is not None:
            raise last_err
        raise ValueError(f"Unable to resolve wikipedia dataset config for ds_name={ds_name!r}")
    return load_dataset(ds_name)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
    "VK_T": SecondMoment_AB_T,
    "KK_T": SecondMoment_AB_T,
    "VV_T": SecondMoment_AB_T,
}


def layer_stats_vk_t(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = _load_hf_text_dataset(ds_name)
        if hasattr(model.config, "n_positions"):
            maxlen = model.config.n_positions
        elif hasattr(model.config, "max_sequence_length"):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, "max_position_embeddings"):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, "seq_length"):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError

        if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
            if hasattr(model.config, "sliding_window") and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 2  # Examine this many dataset texts at once
    if hasattr(model.config, "n_positions"):
        npos = model.config.n_positions
    elif hasattr(model.config, "max_sequence_length"):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
        if hasattr(model.config, "sliding_window") and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
        npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = (
        f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}_vk_t.npz"
    )
    filename = stats_dir / file_extension

    print("Computing VK_T locally....")

    ds = get_ds() if not filename.exists() else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=True, stop=True
                ) as tr:
                    model(**batch)
                feats_in = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats_in = feats_in.to(dtype=dtype)

                model_name_lower = str(model_name).lower()
                if "gpt-j" in model_name_lower:
                    W = nethook.get_parameter(model, f"{layer_name}.weight").to(dtype=dtype)
                    feats_out = feats_in @ W.T
                elif "gpt2" in model_name_lower:
                    W = nethook.get_parameter(model, f"{layer_name}.weight").to(dtype=dtype)
                    feats_out = feats_in @ W
                else:
                    feats_out = flatten_masked_batch(tr.output, batch["attention_mask"])
                    feats_out = feats_out.to(dtype=dtype)

                stat.add(feats_out, feats_in)

    return stat


def layer_stats_kk_t(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = _load_hf_text_dataset(ds_name)
        if hasattr(model.config, "n_positions"):
            maxlen = model.config.n_positions
        elif hasattr(model.config, "max_sequence_length"):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, "max_position_embeddings"):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, "seq_length"):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError

        if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
            if hasattr(model.config, "sliding_window") and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 2  # Examine this many dataset texts at once
    if hasattr(model.config, "n_positions"):
        npos = model.config.n_positions
    elif hasattr(model.config, "max_sequence_length"):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
        if hasattr(model.config, "sliding_window") and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
        npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = (
        f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}_kk_t.npz"
    )
    filename = stats_dir / file_extension

    print("Computing KK_T locally....")

    ds = get_ds() if not filename.exists() else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats_in = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats_in = feats_in.to(dtype=dtype)

                stat.add(feats_in, feats_in)

    return stat


def layer_stats_vv_t(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = _load_hf_text_dataset(ds_name)
        if hasattr(model.config, "n_positions"):
            maxlen = model.config.n_positions
        elif hasattr(model.config, "max_sequence_length"):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, "max_position_embeddings"):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, "seq_length"):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError

        if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
            if hasattr(model.config, "sliding_window") and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 2  # Examine this many dataset texts at once
    if hasattr(model.config, "n_positions"):
        npos = model.config.n_positions
    elif hasattr(model.config, "max_sequence_length"):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
        if hasattr(model.config, "sliding_window") and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
        npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = (
        f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}_vv_t.npz"
    )
    filename = stats_dir / file_extension

    print("Computing VV_T locally....")

    ds = get_ds() if not filename.exists() else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=True, stop=True
                ) as tr:
                    model(**batch)
                feats_in = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats_in = feats_in.to(dtype=dtype)

                model_name_lower = str(model_name).lower()
                if "gpt-j" in model_name_lower:
                    W = nethook.get_parameter(model, f"{layer_name}.weight").to(dtype=dtype)
                    feats_out = feats_in @ W.T
                elif "gpt2" in model_name_lower:
                    W = nethook.get_parameter(model, f"{layer_name}.weight").to(dtype=dtype)
                    feats_out = feats_in @ W
                else:
                    feats_out = flatten_masked_batch(tr.output, batch["attention_mask"])
                    feats_out = feats_out.to(dtype=dtype)

                stat.add(feats_out, feats_out)

    return stat
