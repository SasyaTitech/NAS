from .encore_hparams import ENCOREHyperParams


def apply_encore_to_model(*args, **kwargs):
    from .encore_main import apply_encore_to_model as _apply_encore_to_model

    return _apply_encore_to_model(*args, **kwargs)


__all__ = ["ENCOREHyperParams", "apply_encore_to_model"]

