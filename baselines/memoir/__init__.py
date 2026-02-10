from .memoir_hparams import MEMOIRHyperParams


def apply_memoir_to_model(*args, **kwargs):
    from .memoir_main import apply_memoir_to_model as _apply_memoir_to_model

    return _apply_memoir_to_model(*args, **kwargs)


__all__ = ["MEMOIRHyperParams", "apply_memoir_to_model"]

