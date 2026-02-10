from .NAS_hparams import NASHyperParams


def apply_NAS_to_model(*args, **kwargs):
    from .NAS_main import apply_NAS_to_model as _apply_NAS_to_model

    return _apply_NAS_to_model(*args, **kwargs)


__all__ = ["NASHyperParams", "apply_NAS_to_model"]

