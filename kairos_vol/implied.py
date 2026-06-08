"""隐含波动率求解：Newton + 二分保护，标量与向量两个入口。"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from .core import black76, intrinsic

VOL_FLOOR = 1e-6
VOL_CAP = 5.0


def implied_vol(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    discount: float = 1.0,
    kind: str = "call",
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """由期权价格反解 Black-76 隐含波动率。

    用 Brenner-Subrahmanyam 近似作初值，Newton 迭代 + 二分区间保护，
    对深度实值/虚值也稳定。价格超出无套利区间时抛 :class:`ValueError`。
    """
    price = float(price)
    forward, strike, maturity = float(forward), float(strike), float(maturity)
    discount = float(discount)
    if maturity <= 0:
        raise ValueError("隐含波动率需要 maturity > 0")
    if not price > 0:
        raise ValueError(f"期权价格必须为正，收到 {price!r}")

    lower = intrinsic(forward, strike, kind, discount)
    upper = discount * (forward if str(kind).lower() == "call" else strike)
    if not (lower - 1e-10 <= price <= upper + 1e-10):
        raise ValueError(
            f"价格 {price:.8f} 超出无套利区间 [{lower:.8f}, {upper:.8f}]"
            f"（F={forward:.4f}, K={strike:.4f}, T={maturity:.4f}, {kind}）")
    if price - lower <= 1e-12:
        return VOL_FLOOR

    # Brenner-Subrahmanyam 初值：σ ≈ √(2π/T) · price / (F·df)
    scale = max(discount * forward, 1e-12)
    guess = math.sqrt(2.0 * math.pi / maturity) * (price / scale)
    lo, hi = VOL_FLOOR, VOL_CAP
    sigma = min(max(guess, 2.0 * VOL_FLOOR), 0.5 * (lo + hi) if hi > 1 else hi - 1e-9)

    model = black76(forward, strike, maturity, sigma, discount, kind)["price"]
    for _ in range(int(max_iter)):
        diff = model - price
        if abs(diff) <= tol * max(1.0, price):
            return sigma
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        vega = black76(forward, strike, maturity, sigma, discount, kind)["vega"]
        if vega > 1e-14:
            candidate = sigma - diff / vega
        else:
            candidate = 0.5 * (lo + hi)
        if not (lo < candidate < hi) or not math.isfinite(candidate):
            candidate = 0.5 * (lo + hi)
        sigma = candidate
        model = black76(forward, strike, maturity, sigma, discount, kind)["price"]
        if hi - lo < tol:
            break
    return float(sigma)


def implied_vol_grid(
    prices: Sequence[float],
    forwards: Sequence[float],
    strikes: Sequence[float],
    maturities: Sequence[float],
    discounts: Optional[Sequence[float]] = None,
    kinds: Optional[Iterable[str]] = None,
    *,
    errors: str = "nan",
) -> np.ndarray:
    """向量化反解；``errors="nan"`` 时把失败点记为 NaN（便于做曲面清洗）。"""
    prices = np.asarray(prices, dtype="float64")
    n = prices.size
    forwards = np.broadcast_to(np.asarray(forwards, dtype="float64"), (n,))
    strikes = np.broadcast_to(np.asarray(strikes, dtype="float64"), (n,))
    maturities = np.broadcast_to(np.asarray(maturities, dtype="float64"), (n,))
    if discounts is None:
        discounts = np.ones(n)
    discounts = np.broadcast_to(np.asarray(discounts, dtype="float64"), (n,))
    kind_list = list(kinds) if kinds is not None else ["call"] * n
    if len(kind_list) != n:
        raise ValueError("kinds 长度必须与 prices 一致")

    out = np.full(n, np.nan)
    for i in range(n):
        try:
            out[i] = implied_vol(prices[i], forwards[i], strikes[i], maturities[i],
                                 discounts[i], kind_list[i])
        except (ValueError, ZeroDivisionError):
            if errors == "raise":
                raise
    return out


def total_variance(iv: float, maturity: float) -> float:
    """总方差 ``w = σ²T``。"""
    return float(iv) ** 2 * float(maturity)


__all__ = ["VOL_CAP", "VOL_FLOOR", "implied_vol", "implied_vol_grid", "total_variance"]