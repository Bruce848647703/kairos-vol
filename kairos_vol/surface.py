"""波动率曲面：报价 → 逐到期日 SVI → 总方差沿期限线性插值。

约定
----
- 对数货币度 ``k = ln(K / F)``，总方差 ``w = σ²·T``；
- 同一期限的多个行权价拟合成一条 :class:`~kairos_vol.svi.SVISlice`；
- 不同期限之间在**固定 k** 上对 ``w`` 线性插值（保证不引入负方差），
  超出端点时平坦外推。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .svi import SVISlice


@dataclass
class Quote:
    """一条期权报价（以隐含波动率给出）。"""

    maturity: float
    strike: float
    iv: float
    forward: float
    weight: float = 1.0
    kind: str = "call"

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError("maturity 必须为正")
        if self.strike <= 0 or self.forward <= 0:
            raise ValueError("strike 与 forward 必须为正")
        if not 0 < self.iv < 5:
            raise ValueError("iv 必须在 (0, 5) 内")
        self.maturity = float(self.maturity)
        self.strike = float(self.strike)
        self.forward = float(self.forward)
        self.iv = float(self.iv)
        self.weight = float(self.weight)

    @property
    def moneyness(self) -> float:
        return float(np.log(self.strike / self.forward))

    @property
    def total_variance(self) -> float:
        return self.iv ** 2 * self.maturity

    def to_dict(self) -> Dict[str, float]:
        return {"maturity": self.maturity, "strike": self.strike, "iv": self.iv,
                "forward": self.forward, "weight": self.weight, "kind": self.kind}


class VolSurface:
    """由若干 SVI 切片组成的波动率曲面。"""

    def __init__(self, maturities: Sequence[float], slices: Sequence[SVISlice],
                 forwards: Sequence[float]) -> None:
        if not (len(maturities) == len(slices) == len(forwards)):
            raise ValueError("maturities / slices / forwards 长度必须一致")
        if not maturities:
            raise ValueError("至少需要一个到期日")
        order = np.argsort(np.asarray(maturities, dtype="float64"))
        self.maturities = [float(maturities[i]) for i in order]
        self.slices = [slices[i] for i in order]
        self.forwards = [float(forwards[i]) for i in order]
        if len(set(self.maturities)) != len(self.maturities):
            raise ValueError("maturities 必须唯一")

    # ------------------------------------------------------------------ 构造
    @classmethod
    def fit(cls, quotes: Iterable[Quote]) -> "VolSurface":
        """按期限分组拟合 SVI。"""
        groups: Dict[float, List[Quote]] = {}
        for quote in quotes:
            groups.setdefault(round(float(quote.maturity), 12), []).append(quote)
        if not groups:
            raise ValueError("没有有效报价")

        maturities, slices, forwards = [], [], []
        for maturity in sorted(groups):
            items = groups[maturity]
            k = np.array([q.moneyness for q in items], dtype="float64")
            w = np.array([q.total_variance for q in items], dtype="float64")
            wt = np.array([q.weight for q in items], dtype="float64")
            forward = float(np.mean([q.forward for q in items]))
            if k.size < 5:
                # 点太少时用一个退化的平微笑（b→0），避免过拟合
                atm = float(np.mean(w))
                slices.append(SVISlice(a=atm, b=1e-6, rho=0.0, m=0.0, sigma=0.1))
            else:
                slices.append(SVISlice.fit(k, w, wt))
            maturities.append(float(maturity))
            forwards.append(forward)
        return cls(maturities, slices, forwards)

    # ------------------------------------------------------------------ 查询
    @property
    def n_slices(self) -> int:
        return len(self.slices)

    def forward(self, maturity: float) -> float:
        """按期限线性插值远期价格（仅供展示，曲面本身以 k 为单位）。"""
        return float(np.interp(float(maturity), self.maturities, self.forwards))

    def total_variance(self, k, maturity: float):
        """固定 k，对期限线性插值总方差（标量或数组）。"""
        k_arr = np.asarray(k, dtype="float64")
        t = float(maturity)
        maturities = np.asarray(self.maturities, dtype="float64")
        if t <= maturities[0]:
            return np.asarray(self.slices[0].w(k_arr), dtype="float64")
        if t >= maturities[-1]:
            return np.asarray(self.slices[-1].w(k_arr), dtype="float64")
        upper = int(np.searchsorted(maturities, t))
        lower = upper - 1
        t0, t1 = maturities[lower], maturities[upper]
        w0 = np.asarray(self.slices[lower].w(k_arr), dtype="float64")
        w1 = np.asarray(self.slices[upper].w(k_arr), dtype="float64")
        weight = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (1.0 - weight) * w0 + weight * w1

    def iv(self, k, maturity: float):
        """对数货币度 k 处的隐含波动率。"""
        w = np.maximum(np.asarray(self.total_variance(k, maturity), dtype="float64"), 1e-16)
        return np.sqrt(w / float(maturity))

    def iv_at(self, strike, maturity: float, forward: Optional[float] = None):
        """按行权价取隐含波动率；forward 缺省时用曲面插值的远期。"""
        fwd = self.forward(maturity) if forward is None else float(forward)
        k = np.log(np.asarray(strike, dtype="float64") / fwd)
        return self.iv(k, maturity)

    def atm_vol(self, maturity: float) -> float:
        """k=0 处的隐含波动率。"""
        return float(self.iv(0.0, maturity))

    def atm_total_variance(self, maturity: float) -> float:
        return float(np.asarray(self.total_variance(0.0, maturity)))

    def grid(self, maturities: Optional[Sequence[float]] = None,
             k_range: Tuple[float, float] = (-0.6, 0.6), n_k: int = 41) -> Dict[str, Any]:
        """生成给绘图/表格用的 (maturities × k) 网格。"""
        ts = list(maturities) if maturities is not None else list(self.maturities)
        ks = np.linspace(k_range[0], k_range[1], int(n_k))
        z = np.array([[float(self.iv(k, t)) for k in ks] for t in ts])
        return {"maturities": ts, "k": ks.tolist(), "iv": z.tolist()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maturities": self.maturities,
            "forwards": self.forwards,
            "slices": [{"maturity": t, **s.to_dict()} for t, s in zip(self.maturities, self.slices)],
            "atm_vols": [self.atm_vol(t) for t in self.maturities],
        }


def quotes_from_grid(forwards: Sequence[float], maturities: Sequence[float],
                     strikes: Sequence[float], ivs: Sequence[Sequence[float]]) -> List[Quote]:
    """把二维 IV 网格转成报价列表（ivs 形状 = len(maturities) × len(strikes)）。"""
    out: List[Quote] = []
    for i, maturity in enumerate(maturities):
        for j, strike in enumerate(strikes):
            out.append(Quote(float(maturity), float(strike), float(ivs[i][j]), float(forwards[i])))
    return out


__all__ = ["Quote", "VolSurface", "quotes_from_grid"]