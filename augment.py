# nuevo archivo: lsm_toolkit/augment.py
from tsaug import AddNoise, TimeWarp, Drift

AUG_PIPE = (
    AddNoise(scale=0.02)              # jitter  ≈2 %
    + TimeWarp(max_speed_ratio=1.1)   # ±10 % duración
    + Drift(max_drift=(0.05, 0.05))   # deriva lenta
)
