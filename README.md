# Norm-Anchor Scaling (NAS)

<p align="center">
  <img src="github.png" alt="Norm-Anchor Scaling overview" width="900">
</p>

This repository contains the code for:

**Norm Anchors Make Model Edits Last**

- arXiv: https://arxiv.org/abs/2602.02543
- Method: Norm-Anchor Scaling (NAS), a plug-in stabilizer for sequential Locate-and-Edit (L&E) model editing.

NAS rescales each solved value vector to an original-model reference norm before writing it into the model. The repository includes NAS, baseline editors, dataset wrappers, evaluation scripts, and lightweight tests used for the paper experiments.

## Repository Layout

- `NAS/`: NAS implementation.
- `baselines/`: baseline editing methods, including MEMIT, ROME, FT, MEND, NSE, AlphaEdit, ENCORE, MEMOIR, and LyapLock.
- `experiments/evaluate.py`: main experiment runner.
- `experiments/resume_run.py`: resume supported long sequential-editing runs.
- `experiments/summarize.py`: summarizer for default `results/<METHOD>/run_###` outputs.
- `experiments/smoke_test.py`: tiny end-to-end smoke-test driver.
- `experiments/benchmark_*.py`: runtime and anchor-statistics ablations.
- `dsets/`: dataset wrappers.
- `glue_eval/`: general-capability evaluation utilities and bundled subsets.
- `hparams/`: method/model hyperparameter files.
- `tests/`: lightweight unit tests.

## Installation

Use Python 3.10 or newer. A CUDA GPU is recommended for model-editing runs.

```bash
conda create -n nas python=3.10 -y
conda activate nas
```

Install PyTorch for your CUDA/runtime environment following the official PyTorch instructions, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For gated Hugging Face models such as Llama-3, authenticate before running:

```bash
huggingface-cli login
```

## Paths

Default paths are configured in `globals.yml` and are relative to this repository:

- `results/`: experiment outputs.
- `data/`: downloaded or manually placed datasets.
- `data/stats/`: covariance/statistics cache.
- `data/kvs/`: key/value cache.
- `hparams/`: hyperparameter files.

You can override them with environment variables:

```bash
export NAS_RESULTS_DIR=/path/to/results
export NAS_DATA_DIR=/path/to/data
export NAS_STATS_DIR=/path/to/stats
export NAS_KV_DIR=/path/to/kvs
export NAS_HPARAMS_DIR=/path/to/hparams
```

`NAS_REMOTE_ROOT_URL` controls the remote mirror used for CounterFact, ZsRE, and statistics downloads. The default mirror is `https://memit.baulab.info`.

## Data and Models

Model weights are not included. The provided configs cover:

- `gpt2-xl`
- `EleutherAI/gpt-j-6b`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

Dataset handling:

- `mcf` / `cf`: MultiCounterFact / CounterFact, downloaded from the MEMIT public data mirror.
- `zsre`: downloaded from the MEMIT public data mirror.
- `wikibigedit`: loaded through Hugging Face `datasets` from `lukasthede/WikiBigEdit`.
- `mquake`: place `MQuAKE-CF-3k-v2.json` in `data/` before running.
- GLUE-style general-capability subsets are bundled under `glue_eval/dataset/`.
- MEMOIR auxiliary features are bundled under `data/memoir/`.

For local model mirrors, pass the local path with `--model_name` while keeping the matching hyperparameter file.

## Quick Checks

Run unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run one small NAS edit on GPT-2 XL and MultiCounterFact:

```bash
python -m experiments.evaluate \
  --alg_name NAS \
  --model_name gpt2-xl \
  --hparams_fname gpt2-xl.json \
  --ds_name mcf \
  --dataset_size_limit 1 \
  --skip_generation_tests \
  --generation_test_interval -1 \
  --run_dir_override results/quick_nas_gpt2xl_mcf
```

The final checkpoint summary is written under:

```text
results/quick_nas_gpt2xl_mcf/checkpoint_evals/after_1/summary.json
```

## Running Experiments

The main entry point is:

```bash
python -m experiments.evaluate \
  --alg_name <METHOD> \
  --model_name <HF_MODEL_OR_LOCAL_PATH> \
  --hparams_fname <HPARAMS_JSON> \
  --ds_name <DATASET>
```

Supported methods:

```text
NAS, MEMIT, MEMIT_seq, MEMIT_rect, MEMIT_prune, ROME, FT, MEND,
NSE, AlphaEdit, ENCORE, MEMOIR, LyapLock
```

Supported datasets:

```text
mcf, cf, zsre, mquake, wikibigedit
```

Common options:

- `--dataset_size_limit N`: use only the first `N` records.
- `--num_edits N`: edit `N` records per batch.
- `--checkpoint_eval_interval N`: evaluate after every `N` edit batches.
- `--skip_generation_tests`: skip slow generation-based tests.
- `--generation_test_interval -1`: disable generation tests.
- `--use_cache`: reuse cached key/value statistics when available.
- `--run_dir_override PATH`: write outputs to a fixed directory.
- `--hparam key=value`: override a top-level hyperparameter; repeat as needed.

## Example Commands

NAS on GPT-J with MultiCounterFact:

```bash
python -m experiments.evaluate \
  --alg_name NAS \
  --model_name EleutherAI/gpt-j-6b \
  --hparams_fname EleutherAI_gpt-j-6B.json \
  --ds_name mcf \
  --dataset_size_limit 1000 \
  --num_edits 1 \
  --checkpoint_eval_interval 100 \
  --skip_generation_tests \
  --generation_test_interval -1 \
  --run_dir_override results/nas_gptj_mcf_1k
```

MEMIT baseline with the same model and dataset:

```bash
python -m experiments.evaluate \
  --alg_name MEMIT \
  --model_name EleutherAI/gpt-j-6b \
  --hparams_fname EleutherAI_gpt-j-6B.json \
  --ds_name mcf \
  --dataset_size_limit 1000 \
  --num_edits 1 \
  --checkpoint_eval_interval 100 \
  --skip_generation_tests \
  --generation_test_interval -1 \
  --run_dir_override results/memit_gptj_mcf_1k
```

NAS on WikiBigEdit:

```bash
python -m experiments.evaluate \
  --alg_name NAS \
  --model_name EleutherAI/gpt-j-6b \
  --hparams_fname EleutherAI_gpt-j-6B.json \
  --ds_name wikibigedit \
  --dataset_size_limit 1000 \
  --num_edits 1 \
  --checkpoint_eval_interval 100 \
  --wikibigedit_checkpoint_eval_sample_ratio 0.1 \
  --skip_generation_tests \
  --generation_test_interval -1 \
  --run_dir_override results/nas_gptj_wikibigedit_1k
```

For full long-horizon runs, increase or remove `--dataset_size_limit`. Paper-scale runs require substantially more GPU time because they repeatedly edit large language models and evaluate checkpoints.

## Outputs

By default, runs are written to `results/<METHOD>/run_###/`. If `--run_dir_override` is set, outputs are written to that path instead.

Important files:

- `params.json`: exact hyperparameters used by the run.
- `hparam_overrides.json`: hyperparameter overrides, if `--hparam` was used.
- `checkpoint_evals/after_<N>/results.jsonl`: per-case checkpoint results after `N` edited records.
- `checkpoint_evals/after_<N>/summary.json`: aggregated checkpoint metrics.
- `resume/state.json`: resume metadata for supported algorithms.
- `edited_weights/after_<N>/rewrite_module_weights.pt`: optional saved rewrite-module weights.
- `glue_eval/*.json`: optional downstream/general-capability evaluation outputs.

To inspect a checkpoint summary:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("results/nas_gptj_mcf_1k/checkpoint_evals/after_1000/summary.json")
print(json.dumps(json.load(open(p))["metrics"], indent=2))
PY
```

To summarize runs stored in the default layout, for example `results/NAS/run_###`:

```bash
python -m experiments.summarize --dir_name NAS
```

## Resuming Runs

Supported algorithms write resumable state under `run_dir/resume/`. To continue an existing run:

```bash
python -m experiments.resume_run \
  --run_dir results/NAS/run_000 \
  --add_cases 1000
```

`--continue_from_run` reuses an existing run directory and hyperparameters in the main evaluator. Use `experiments.resume_run` when you want to load the saved resume state and add more cases.

## Smoke-Test Driver

`experiments/smoke_test.py` runs tiny `dataset_size_limit=1` cases across method and hyperparameter combinations. It is intended for functional checks, not for reproducing paper tables.

```bash
python -m experiments.smoke_test --out_root smoke_reports/basic
```

If a model should resolve to a local path, edit `smoke_model_map.json` or pass a custom map:

```bash
python -m experiments.smoke_test \
  --model_map smoke_model_map.json \
  --out_root smoke_reports/local_models
```

## Compute Notes

All paper experiments were run on NVIDIA H100 GPUs. L&E-style methods on GPT-J, Llama-3, and Qwen2.5 generally require high-memory GPUs; if memory is limited, start with `gpt2-xl`, `--dataset_size_limit 1`, and generation tests disabled.

## Licenses and Terms

This code uses externally hosted models, datasets, and baseline implementations. Users should follow the licenses, model cards, dataset cards, and terms of use provided by the original authors or hosting platforms.

- Model weights are not redistributed.
- Gated models such as Llama-3 require the corresponding access approval.
- CounterFact and ZsRE are downloaded from the MEMIT public data mirror.
- WikiBigEdit is loaded from Hugging Face `datasets`.
- Baseline editor code and hyperparameters follow the original authors' releases when available, as cited in the paper.

## Citation

If you use this repository, please cite the arXiv preprint:

https://arxiv.org/abs/2602.02543

The arXiv metadata may take some time to reflect the latest replacement. Once it is updated, please use the BibTeX exported from the arXiv page.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
