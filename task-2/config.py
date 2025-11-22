"""
Central place for experiment-wide settings (seeds, filesystem layout, device selection).
Keeping this lightweight and documented helps the other modules stay focused on ML logic.
"""
import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Reproducibility: every module imports SEED so train/test splitting and sampling stay stable.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch

    # Autodetect GPU but gracefully fall back to CPU, so the pipeline still runs on laptops.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # pragma: no cover
    DEVICE = "cpu"

# Resolve project-relative paths once, so the rest of the code does not need to worry about cwd.
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TASK_DIR)
DATA_DIR = os.path.abspath(
    os.environ.get("RATING_DATA_DIR", os.path.join(REPO_ROOT, "engagetest", "data"))
)
CACHE_DIR = os.path.abspath(
    os.environ.get("RATING_CACHE_DIR", os.path.join(TASK_DIR, "cache"))
)
os.makedirs(CACHE_DIR, exist_ok=True)

