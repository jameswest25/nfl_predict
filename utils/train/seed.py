# utils/train/seed.py
import hashlib
import os
import random
import numpy as np

def deterministic_seed(base_seed: int, problem_name: str) -> int:
    """
    Create a stable per-problem seed by hashing the problem_name with base_seed.
    Returns a 32-bit int (safe for most ML libs).
    """
    s = f"{base_seed}:{problem_name}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=4).hexdigest()
    return int(h, 16) & 0x7FFFFFFF

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # Optional: if torch installed
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
