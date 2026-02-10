import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise TypeError(f"Expected hyperparameters JSON dict at {fpath}")
        data = dict(data)

        if "nas_restat" in data and "nas_restart" not in data:
            data["nas_restart"] = data.pop("nas_restat")

        if is_dataclass(cls):
            allowed = {f.name for f in fields(cls)}
            extra = sorted(set(data) - allowed)
            if extra:
                for k in extra:
                    data.pop(k)
                print(
                    f"Warning: Dropping unknown hyperparameter keys for {cls.__name__}: {extra}"
                )

        obj = cls(**data)
        obj._hparams_path = str(fpath)
        obj._hparams_name = Path(fpath).name
        return obj
