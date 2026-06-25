"""SSVI 参数化（Gatheral & Jacquier 2014）：整条曲面的全局参数化。

::

    w(k, θ) = θ/2 · [ 1 + ρ·φ(θ)·k + sqrt((φ(θ)·k + ρ)² + (1 - ρ²)) ]

其中 θ 是 ATM 总方差（θ = σ_atm²·T），φ(θ) = η / θ^γ。只需 **3 个全局参数**
``(ρ, η, γ)`` 就能描述整个曲面，天然满足大部分无套利约束：

- ``|ρ| < 1``
- ``η(1 + |ρ|) ≤ 2``（Durrleman 条件的充分条件）
- ``0 < γ < 1/2``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .numerics import nelder_mead, weighted_sse


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


@dataclass
class SSVISurface:
    """SSVI 全局曲面参数。"""

    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        if abs(self.rho) >= 1.0:
            raise ValueError("rho 必须在 (-1, 1) 内")
        if not self.eta > 0:
            raise ValueError("eta 必须为正")
        if not 0.0 < self.gamma < 0.5:
            raise ValueError("gamma 必须在 (0, 0.5) 内（无套利条件）")

    def phi(self, theta: float) -> float:
        """φ(θ) = η / θ^γ。"""
        return float(self.eta) / float(theta) ** float(self.gamma)

    def w(self, k, theta: float):
        """给定 ATM 总方差 θ，返回对数货币度 k 处的总方差。"""
        k = np.asarray(k, dtype="float64")
        phi = self.phi(theta)
        rho = float(self.rho)
        inner = (phi * k + rho) ** 2 + (1.0 - rho * rho)
        return 0.5 * float(theta) * (1.0 + rho * phi * k + np.sqrt(inner))

    def iv(self, k, theta: float, maturity: float):
        return np.sqrt(np.maximum(np.asarray(self.w(k, theta), dtype="float64"), 0.0) / float(maturity))

    def no_arb_ok(self) -> bool:
        """SSVI 的充分无套利条件：η(1+|ρ|) ≤ 2 且 0 < γ < 1/2。"""
        return self.eta * (1.0 + abs(self.rho)) <= 2.0 + 1e-12

    def to_dict(self) -> Dict[str, float]:
        return {"rho": self.rho, "eta": self.eta, "gamma": self.gamma}

    # ------------------------------------------------------------------ 拟合
    @classmethod
    def fit(
        cls,
        slices: Sequence[Tuple[float, Sequence[float], Sequence[float]]],
        weights: Optional[Sequence[Sequence[float]]] = None,
        *,
        max_iter: int = 4000,
    ) -> "SSVISurface":
        """联合拟合多个到期日。

        参数
        ----
        slices: 序列，每项 ``(atm_total_variance, k_array, w_array)``；
        weights: 与 ``slices`` 同结构的可选权重。
        """
        if not slices:
            raise ValueError("至少需要一个到期日")
        prepared: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        for idx, (theta, k, w) in enumerate(slices):
            k_arr = np.asarray(k, dtype="float64").ravel()
            w_arr = np.asarray(w, dtype="float64").ravel()
            if k_arr.size != w_arr.size:
                raise ValueError("k 与 w 长度必须一致")
            if weights is None:
                wts = np.ones_like(k_arr)
            else:
                wts = np.asarray(weights[idx], dtype="float64").ravel()
                wts = np.broadcast_to(wts, k_arr.shape).copy()
            prepared.append((float(theta), k_arr, w_arr, wts))

        def unpack(x: np.ndarray) -> "SSVISurface":
            return cls(rho=float(np.tanh(x[0])),
                       eta=float(np.exp(x[1])),
                       gamma=float(0.5 * _sigmoid(x[2])))

        def objective(x: np.ndarray) -> float:
            surf = unpack(x)
            total = 0.0
            for theta, k_arr, w_arr, wts in prepared:
                total += weighted_sse(surf.w(k_arr, theta), w_arr, wts)
            if not surf.no_arb_ok():
                total += 1e3 * (surf.eta * (1.0 + abs(surf.rho)) - 2.0) ** 2
            return float(total)

        x0 = np.array([np.arctanh(-0.5), np.log(1.0), _logit(0.4)])
        x_best, _ = nelder_mead(objective, x0, step=0.05, max_iter=max_iter, tol=1e-14)
        return unpack(x_best)


__all__ = ["SSVISurface"]