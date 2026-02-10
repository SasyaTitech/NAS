from .AlphaEdit_hparams import AlphaEditHyperParams


def apply_AlphaEdit_to_model(*args, **kwargs):
    from .AlphaEdit_main import apply_AlphaEdit_to_model as _apply_AlphaEdit_to_model

    return _apply_AlphaEdit_to_model(*args, **kwargs)


__all__ = ["AlphaEditHyperParams", "apply_AlphaEdit_to_model"]
