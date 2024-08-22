import torch
from contextlib import suppress
from functools import partial


def get_autocast(precision):
    if precision == "amp":
        return partial(torch.amp.autocast, "cuda") if torch.cuda.is_available() else partial(torch.amp.autocast, "cpu")
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return lambda: torch.amp.autocast(device, dtype=torch.bfloat16)
    else:
        return suppress
