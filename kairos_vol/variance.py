"""方差互换与波动率 carry：从曲面算公允方差、已实现方差与 carry。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .core import black76
from .surface import VolSurface

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def fair_variance_strike(surface: VolSurface, maturity: float,
                         k_range=(-3.0, 3.0), n: int = 601) -> float:
    """对数合约复制法（forward 口径，贴现因子=1）::

        E[Var] = 2/T · [ ∫_{K<F} P(K)/K² dK + ∫_{K>F} C(K)/K² dK ]
    """
    maturity = float(maturity)
    if maturity <= 0:
        raise ValueError("maturity 必须为正")
    forward = surface.forward(maturity)
    ks = np.linspace(k_range[0], k_range[1], int(n))
    strikes = forward * np.exp(ks)
    ivs = np.asarray(surface.iv(ks, maturity), dtype="float64")
    prices = np.empty_like(strikes)
    for i, (strike, iv) in enumerate(zip(strikes, ivs)):
        kind = "put" if strike < forward else "call"
        prices[i] = black76(forward, float(strike), maturity, float(iv), 1.0, kind)["price"]
    integral = float(_trapz(prices / strikes ** 2, strikes))
    return max(2.0 * integral / maturity, 0.0)


def fair_vol_strike(surface: VolSurface, maturity: float, **kwargs: Any) -> float:
    """方差互换的公允波动率（= sqrt(公允方差)）。"""
    return float(np.sqrt(fair_variance_strike(surface, maturity, **kwargs)))


def realized_variance(prices: Sequence[float], periods_per_year: float = 252.0,
                      log_returns: bool = True) -> float:
    """已实现方差（年化）。"""
    p = np.asarray(prices, dtype="float64").ravel()
    if p.size < 2:
        raise ValueError("至少需要 2 个价格点")
    if np.any(p <= 0):
        raise ValueError("价格必须为正")
    returns = np.diff(np.log(p)) if log_returns else np.diff(p) / p[:-1]
    return float(np.sum(returns ** 2) * float(periods_per_year) / returns.size)


def variance_swap_pnl(realized_var: float, strike_var: float,
                      variance_notional: float = 1.0) -> float:
    """方差互换买方盈亏：``名义本金 × (已实现方差 - 行权方差)``。"""
    return float(variance_notional) * (float(realized_var) - float(strike_var))


def vol_carry(surface: VolSurface, maturity: float) -> Dict[str, float]:
    """ATM 隐含方差与公允方差之差（carry 的近似）。"""
    atm_vol = float(surface.atm_vol(maturity))
    fair_var = fair_variance_strike(surface, maturity)
    atm_var = atm_vol ** 2
    return {
        "maturity": float(maturity),
        "atm_vol": atm_vol,
        "atm_variance": atm_var,
        "fair_variance": fair_var,
        "fair_vol": float(np.sqrt(fair_var)),
        "carry_variance": fair_var - atm_var,
        "carry_vol": float(np.sqrt(fair_var)) - atm_vol,
    }


def term_structure(surface: VolSurface,
                   maturities: Optional[Sequence[float]] = None) -> List[Dict[str, float]]:
    """逐期限的 ATM / 公允波动率 / carry 表。"""
    ts = list(maturities) if maturities is not None else list(surface.maturities)
    return [vol_carry(surface, float(t)) for t in ts]


__all__ = [
    "fair_variance_strike",
    "fair_vol_strike",
    "realized_variance",
    "term_structure",
    "variance_swap_pnl",
    "vol_carry",
]