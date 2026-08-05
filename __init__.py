"""ComfyUI-MiniMaxH3: streaming DiT node pack with ring-buffer BlockSwap.

Layout (BerniniRWrapper-style, by function):

* ``models/``  - neural nets + quantized weights (dit, vae, quant, lora)
* ``utils/``   - runtime infrastructure (types, config, stream, blockswap,
                 lifecycle, teacache, injection, encoder_use)
* ``nodes/``   - thin ComfyUI shells over the runtime
* ``tests/``   - equivalence / memory / release test suite

The encoder runtime is vendored under ``models/text_encoder`` and is loaded
from the configured model directory at runtime.
"""

from . import models
from . import utils

WEB_DIRECTORY = "./web"

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # outside ComfyUI (tests) nodes need folder_paths
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["models", "utils", "WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
