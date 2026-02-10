# NAS: Public Code Release

This repository contains the code used in the accompanying paper and is organized
for public release and reproduction.

## Paper

- Title: Toward Ultra-Long-Horizon Sequential Model Editing
- arXiv: https://arxiv.org/abs/2602.02543

## Setup

- Recommended Python: 3.10+
- Install PyTorch following the official instructions for your platform.
- Install remaining deps:

```bash
pip install -r requirements.txt
```

## Quick sanity check

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Running experiments (`experiments/evaluate.py`)

All main experiments are run via the CLI entry in `experiments/evaluate.py`:

```bash
python -m experiments.evaluate \
  --alg_name <ALG> \
  --model_name <HF_ID_OR_LOCAL_PATH> \
  --hparams_fname <FILE.json> \
  --ds_name <DATASET>
```

### Key arguments

- `--alg_name`: editing algorithm. Options include `NAS`, `AlphaEdit`, `ENCORE`, `MEMOIR`, `ROME`, `MEMIT*`, `FT`, `MEND`, `NSE`, `LyapLock`.
- `--model_name`: HuggingFace model id or a local model path.
- `--hparams_fname`: hyperparameters JSON.
  - For `MEMIT*` variants (`MEMIT`, `MEMIT_seq`, `MEMIT_rect`, `MEMIT_prune`), hparams are loaded from `hparams/MEMIT/`.
  - For other algs, hparams are loaded from `hparams/<ALG>/`.
- `--ds_name`: dataset (`mcf`, `cf`, `zsre`, `mquake`, `wikibigedit`).

### Common options

- `--dataset_size_limit N`: run only the first `N` cases (useful for debugging).
- `--num_edits K`: edit `K` cases simultaneously per step (not supported for `--ds_name cf`).
- `--skip_generation_tests`: disable slow generation-based tests (runs only probability / token-accuracy tests).
- `--generation_test_interval N`: run generation tests every `N` edited records (`-1` disables). Note: generation tests are forced off for `MEMOIR` and `wikibigedit`.
- `--checkpoint_eval_interval N`: run a full checkpoint evaluation every `N` edit *batches* (each batch edits `--num_edits` records). `0` means “only final checkpoint”.
- `--save_edited_weights_interval N`: when `--checkpoint_eval_interval>0`, save rewrite-module weights every `N` checkpoint evals.
- `--downstream_eval_steps N`: run GLUE-style downstream evaluation every `N` edit *batches* (writes to `run_dir/glue_eval/`).
- `--use_cache`: load/write cached K/V files under `KV_DIR` (see “Paths / portability” below).
- `--hparam key=value`: override a top-level hyperparameter field from the loaded hparams JSON (repeatable). Example:
  - `--hparam mom2_n_samples=50000 --hparam layers=[13,14,15]`
- `--run_dir_override PATH`: write all artifacts to `PATH` instead of `results/<ALG>/run_###`.

### Resuming runs

`experiments/evaluate.py` writes a resumable state for supported algorithms under `run_dir/resume/`.
To actually resume and add more cases, use:

```bash
python -m experiments.resume_run --run_dir <RUN_DIR> --add_cases <N>
```

(`--continue_from_run` reuses a run directory/hparams, but does not automatically load the resume state.)

## Run a minimal evaluation (example)

This will download required HuggingFace datasets/models on first run. Some methods
also compute/cache statistics and can be slow; a GPU is recommended.

```bash
python -m experiments.evaluate \
  --alg_name NAS \
  --model_name gpt2-xl \
  --hparams_fname gpt2-xl.json \
  --ds_name mcf \
  --dataset_size_limit 1 \
  --skip_generation_tests \
  --run_dir_override results/public_demo
```

## Results: where to look / how to read

### Output directory layout

By default, each run writes to `results/<ALG>/run_###/` (or `--run_dir_override`).
Important files/subfolders:

- `params.json`: the exact hparams used for the run.
- `checkpoint_evals/after_<N>/results.jsonl`: per-case JSONL results for the checkpoint after `N` edited records.
- `checkpoint_evals/after_<N>/summary.json`: aggregated checkpoint summary (means/stds over edited cases).
- `resume/state.json`: resume metadata + pointers to saved tensors (for supported algs).
- `edited_weights/after_<N>/rewrite_module_weights.pt`: optional saved rewrite-module weights (if enabled).
- `glue_eval/*.json`: optional downstream (GLUE-style) eval outputs (if enabled).

### `results.jsonl` format (per edited case)

Each line is a JSON object with (at least) the following keys:

- `case_id`: dataset case id.
- `edit_order_idx`: order in which the case was edited (0-based).
- `requested_rewrite`: the edit request (prompt/subject/target).
- `post`: post-edit evaluation metrics (dataset-dependent; see below).

### `summary.json` metrics (checkpoint-level)

`summary.json` contains a `metrics` dict with aggregated values. Common entries:

- `post_rewrite_acc`, `post_paraphrase_acc`, `post_neighborhood_acc`: mean token-level accuracies (reported as percentages in `summary.json`).
- For CounterFact-style datasets, you may also see:
  - `post_rewrite_success`, `post_paraphrase_success`: fraction of prompts where the edited target is assigned higher probability than the original target.
  - `post_neighborhood_success`: fraction of locality prompts where the original target remains more likely than the edited target.
- For WikiBigEdit, you may see:
  - `post_ES`, `post_GS`, `post_LS`: teacher-forcing token-level accuracies for edit/generalization/locality.

To quickly inspect a checkpoint summary:

```bash
python -c "import json; p='results/<ALG>/run_000/checkpoint_evals/after_1/summary.json'; print(json.load(open(p))['metrics'])"
```

## Paths / portability

You can override common directories via environment variables:

- `NAS_RESULTS_DIR`, `NAS_DATA_DIR`, `NAS_STATS_DIR`, `NAS_HPARAMS_DIR`, `NAS_KV_DIR`
- `NAS_REMOTE_ROOT_URL`

## Compute / hardware notes

- All paper experiments were run on NVIDIA H100 GPUs.
- For Locate-and-Edit (L&E) style methods, our experiments require at least **48 GB** of GPU memory (VRAM). If you have less, start with smaller models (e.g. `gpt2-xl`) and small `--dataset_size_limit`.

## Notes

- This public release excludes large local artifacts such as experiment outputs and
  temporary logs.

## Citation

If you use this repository, please cite:

```bibtex
@misc{liu2026ultralonghorizonsequentialmodelediting,
      title={Toward Ultra-Long-Horizon Sequential Model Editing},
      author={Mingda Liu and Zhenghan Zhu and Ze'an Miao and Katsuki Fujisawa},
      year={2026},
      eprint={2602.02543},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02543},
}
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
