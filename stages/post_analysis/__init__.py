def run(*args, **kwargs):
    from stages.post_analysis.__main__ import run as _run

    return _run(*args, **kwargs)

__all__ = ["run"]
