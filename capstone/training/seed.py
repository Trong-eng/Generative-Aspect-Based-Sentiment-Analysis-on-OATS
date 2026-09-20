"""Seed supported random number generators without making ML imports mandatory."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> int:
    value = int(seed)
    random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(value)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    try:
        from transformers import set_seed
    except ImportError:
        set_seed = None
    if set_seed is not None:
        set_seed(value)
    return value

