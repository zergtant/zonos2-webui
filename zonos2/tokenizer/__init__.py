def tokenize_worker(*args, **kwargs):
    from .server import tokenize_worker as _tokenize_worker

    return _tokenize_worker(*args, **kwargs)

__all__ = ["tokenize_worker"]
