"""数据入口：合成曲面、CSV 读写。

合成数据用 SSVI 生成，因此「真参数」已知，可以用来检验 SVI/SSVI 的校准精度。
"""
from __future__ import annotations

import csv
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .ssvi import SSVISurface
from .surface import Quote


def synthetic_ssvi_quotes(
    *,
    rho: float = -0.6,
    eta: float = 0.8,
    gamma: float = 0.3,
    forward: float = 100.0,
    maturities: Sequence[float] = (0.08, 0.25, 0.5, 1.0, 2.0),
    atm_vol: Union[Sequence[float], Callable[[float], float]] = (0.21, 0.20, 0.185, 0.175, 0.17),
    k_min: float = -0.6,
    k_max: float = 0.6,
    n_k: int = 21,
    noise: float = 0.0,
    seed: int = 7,
    weight: float = 1.0,
) -> List[Quote]:
    """用 SSVI 生成一组「干净」的期权报价（可选加噪声）。"""
    surface = SSVISurface(rho=rho, eta=eta, gamma=gamma)
    rng = np.random.default_rng(seed)
    ks = np.linspace(k_min, k_max, int(n_k))
    quotes: List[Quote] = []
    for idx, maturity in enumerate(maturities):
        t = float(maturity)
        if callable(atm_vol):
            sigma_atm = float(atm_vol(t))
        else:
            atm = np.asarray(atm_vol, dtype="float64").ravel()
            if atm.size == len(maturities):
                sigma_atm = float(atm[idx])
            else:                                   # 自动把给定曲线映射到期限区间
                grid = np.linspace(float(min(maturities)), float(max(maturities)), atm.size)
                sigma_atm = float(np.interp(t, grid, atm))
        theta = sigma_atm ** 2 * t
        w = np.asarray(surface.w(ks, theta), dtype="float64")
        iv = np.sqrt(np.maximum(w / t, 1e-12))
        if noise > 0:
            iv = iv + rng.normal(0.0, float(noise), size=iv.size)
        for k, vol in zip(ks, iv):
            strike = float(forward * np.exp(float(k)))
            quotes.append(Quote(maturity=t, strike=strike, iv=float(vol),
                                forward=float(forward), weight=float(weight)))
    return quotes


def _pick(row: Dict[str, str], keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in keys:
        if key in lowered and str(lowered[key]).strip() != "":
            return str(lowered[key]).strip()
    return default


def load_quotes_csv(path: str) -> List[Quote]:
    """读取报价 CSV。

    必需列：``maturity``（或 ``t``/``tenor``）、``strike``（或 ``k``）、``iv``（或 ``vol``）；
    可选列：``forward``（缺省用 ``spot``，再缺省用 strike 的中位数）、``weight``、``kind``。
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} 没有数据")

    strikes = [float(_pick(r, ("strike", "k", "行权价"), "0")) for r in rows]
    default_forward = float(np.median(strikes)) if strikes else 100.0

    quotes: List[Quote] = []
    for row in rows:
        maturity = float(_pick(row, ("maturity", "t", "tenor", "期限"), "1"))
        strike = float(_pick(row, ("strike", "k", "行权价"), "0"))
        iv = float(_pick(row, ("iv", "vol", "volatility", "隐含波动率"), "0"))
        forward = float(_pick(row, ("forward", "f", "spot", "s"), str(default_forward)))
        weight = float(_pick(row, ("weight", "w"), "1"))
        kind = str(_pick(row, ("kind", "type", "cp"), "call"))
        quotes.append(Quote(maturity, strike, iv, forward, weight, kind))
    return quotes


def save_quotes_csv(quotes: Iterable[Quote], path: str) -> None:
    """把报价写成标准 CSV。"""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["maturity", "strike", "iv", "forward", "weight", "kind"])
        for q in quotes:
            writer.writerow([q.maturity, q.strike, q.iv, q.forward, q.weight, q.kind])


def surface_from_quotes(quotes: Iterable[Quote]):
    """便捷入口：报价 -> 曲面。"""
    from .surface import VolSurface

    return VolSurface.fit(quotes)


__all__ = [
    "load_quotes_csv",
    "save_quotes_csv",
    "surface_from_quotes",
    "synthetic_ssvi_quotes",
]