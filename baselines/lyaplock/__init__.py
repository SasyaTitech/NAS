from .LyapLock_hparams import LyapLockHyperParams


def apply_lyaplock_to_model(*args, **kwargs):
    from .lyaplock_main import apply_lyaplock_to_model as _apply_lyaplock_to_model

    return _apply_lyaplock_to_model(*args, **kwargs)


__all__ = ["LyapLockHyperParams", "apply_lyaplock_to_model"]

