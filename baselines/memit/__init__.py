from .memit_hparams import MEMITHyperParams


def apply_memit_to_model(*args, **kwargs):
    from .memit_main import apply_memit_to_model as _apply_memit_to_model

    return _apply_memit_to_model(*args, **kwargs)


__all__ = ["MEMITHyperParams", "apply_memit_to_model"]
