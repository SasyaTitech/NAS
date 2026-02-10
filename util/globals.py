import os
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_GLOBALS_PATH = _ROOT / "globals.yml"

with open(_GLOBALS_PATH, "r") as stream:
    data = yaml.safe_load(stream)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_ROOT / path).resolve()

def _get_cfg_str(key: str, env_var: str) -> str:
    override = os.environ.get(env_var)
    if override is not None and str(override).strip():
        return str(override).strip()
    val = data.get(key)
    if val is None:
        raise KeyError(f"Missing required key {key!r} in globals.yml")
    if not isinstance(val, str):
        raise TypeError(f"Expected a string for {key!r} in globals.yml; got {type(val)}")
    return val


RESULTS_DIR = _resolve_path(_get_cfg_str("RESULTS_DIR", "NAS_RESULTS_DIR"))
DATA_DIR = _resolve_path(_get_cfg_str("DATA_DIR", "NAS_DATA_DIR"))
STATS_DIR = _resolve_path(_get_cfg_str("STATS_DIR", "NAS_STATS_DIR"))
HPARAMS_DIR = _resolve_path(_get_cfg_str("HPARAMS_DIR", "NAS_HPARAMS_DIR"))
KV_DIR = _resolve_path(_get_cfg_str("KV_DIR", "NAS_KV_DIR"))

REMOTE_ROOT_URL = os.environ.get("NAS_REMOTE_ROOT_URL", data["REMOTE_ROOT_URL"])
