"""ComfyUI interrupt checks for long-running model loops."""

from __future__ import annotations

_model_management = None


def check_interrupt() -> None:
    global _model_management
    if _model_management is None:
        try:
            from comfy import model_management
        except ImportError:
            return
        _model_management = model_management
    _model_management.throw_exception_if_processing_interrupted()
