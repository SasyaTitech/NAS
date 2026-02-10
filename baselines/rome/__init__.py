from .rome_hparams import ROMEHyperParams


def apply_rome_to_model(*args, **kwargs):
    from .rome_main import apply_rome_to_model as _apply_rome_to_model

    return _apply_rome_to_model(*args, **kwargs)


def execute_rome(*args, **kwargs):
    from .rome_main import execute_rome as _execute_rome

    return _execute_rome(*args, **kwargs)


__all__ = ["ROMEHyperParams", "apply_rome_to_model", "execute_rome"]
