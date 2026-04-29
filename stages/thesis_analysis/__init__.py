def run(*args, **kwargs):
    from stages.thesis_analysis.__main__ import run as _run

    return _run(*args, **kwargs)

__all__ = ["run"]
