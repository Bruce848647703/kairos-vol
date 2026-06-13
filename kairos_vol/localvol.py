"""Dupire 局部波动率：从总方差曲面反推 ``σ_loc(K, T)``。

在总方差 ``w(k, T)``（``k = ln(K/F)``）表示下::

    σ_loc² = (∂w/∂T) /
             [ 1 - (k/w)·w_k + (1/4)(-1/4 - 1/w + k²/w²)·w_k² + (1/2)·w_kk ]

其中 ``w_k = ∂w/∂k``、``w_kk = ∂²w/∂k²``。平坦曲面（w = σ²T）退化为常数 σ，
这是最直接的单元测试。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .surface import VolSurface


def dupire_local_variance(surface: VolSurface, k: float, maturity: float,
                          dk: float = 1e-4) -> float:
    """局部方差 σ_loc²（对数货币度 k 处）。"""
    k = float(k)
    maturity = float(maturity)
    w = float(np.asarray(surface.total_variance(k, maturity)))

    w_up = float(np.asarray(surface.total_variance(k + dk, maturity)))
    w_dn = float(np.asarray(surface.total_variance(k - dk, maturity)))
    w_k = (w_up - w_dn) / (2.0 * dk)
    w_kk = (w_up - 2.0 * w + w_dn) / (dk * dk)

    mats = list(surface.maturities)
    if len(mats) > 1 and maturity <= mats[0] + 1e-12:
        t_lo, t_hi = mats[0], mats[1]              # 首端：前向差分
    elif len(mats) > 1 and maturity >= mats[-1] - 1e-12:
        t_lo, t_hi = mats[-2], mats[-1]            # 末端：后向差分
    else:                                          # 中间：用所在区间的割线
        upper = int(np.searchsorted(np.asarray(mats), maturity))
        upper = min(max(upper, 1), len(mats) - 1)
        t_lo, t_hi = mats[upper - 1], mats[upper]
        if t_hi <= t_lo:
            t_lo, t_hi = max(maturity - 1e-3, 1e-8), maturity + 1e-3

    w_lo = float(np.asarray(surface.total_variance(k, t_lo)))
    w_hi = float(np.asarray(surface.total_variance(k, t_hi)))
    w_t = (w_hi - w_lo) / (t_hi - t_lo)

    safe_w = max(w, 1e-14)
    denominator = (1.0
                   - (k / safe_w) * w_k
                   + 0.25 * (-0.25 - 1.0 / safe_w + k * k / (safe_w * safe_w)) * w_k * w_k
                   + 0.5 * w_kk)
    if denominator <= 1e-12 or w_t <= 0:
        return float("nan")
    return float(w_t / denominator)


def dupire_local_vol(surface: VolSurface, k: float, maturity: float,
                     dk: float = 1e-4) -> float:
    """局部波动率 σ_loc（取方差平方根）。"""
    variance = dupire_local_variance(surface, k, maturity, dk)
    return float(np.sqrt(variance)) if np.isfinite(variance) and variance > 0 else float("nan")


def local_vol_grid(surface: VolSurface,
                   maturities: Optional[Sequence[float]] = None,
                   k_range=(-0.5, 0.5), n_k: int = 21) -> Dict[str, Any]:
    """生成网格化的局部波动率（给热力图/表格用）。"""
    ts = list(maturities) if maturities is not None else list(surface.maturities)
    ks = np.linspace(k_range[0], k_range[1], int(n_k))
    values: List[List[float]] = []
    for t in ts:
        values.append([dupire_local_vol(surface, float(k), float(t)) for k in ks])
    return {"maturities": ts, "k": ks.tolist(), "local_vol": values}


__all__ = ["dupire_local_variance", "dupire_local_vol", "local_vol_grid"]