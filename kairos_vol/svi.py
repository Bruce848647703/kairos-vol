"""SVI 参数化（Gatheral 2004）：单个到期日的波动率微笑。

以对数货币度 ``k = ln(K/F)`` 与**总方差** ``w(k) = σ²(k)·T`` 表示::

    w(k) = a + b · ( ρ(k - m) + sqrt((k - m)² + σ²) )

5 个参数：``a``（方差水平）、``b``（斜率幅度）、``ρ``（偏斜）、``m``（平移）、
``σ``（曲率平滑）。无套利条件：

- ``w(k) ≥ 0``  ⇒  ``a + b·σ·sqrt(1-ρ²) ≥ 0``
- 蝶式套利（密度非负）⇔ Durrleman 条件 ``g(k) ≥ 0``（见 :meth:`SVISlice.g`）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .numerics import nelder_mead, weighted_sse


def _as_arrays(k: Sequence[float], w: Sequence[float]):
    k = np.asarray(k, dtype="float64").ravel()
    w = np.asarray(w, dtype="float64").ravel()
    if k.size != w.size:
        raise ValueError("k 与 w 长度必须一致")
    if k.size < 3:
        raise ValueError("SVI 拟合至少需要 3 个点")
    mask = np.isfinite(k) & np.isfinite(w) & (w > -1e-12)
    k, w = k[mask], np.maximum(w[mask], 0.0)
    if k.size < 3:
        raise ValueError("有效点不足 3 个")
    order = np.argsort(k)
    return k[order], w[order]


@dataclass
class SVISlice:
    """一个到期日的 SVI 微笑。"""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        """总方差 w(k)。"""
        return self.a + self.b * (self.rho * (k - self.m)
                                  + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def dw(self, k):
        """w'(k)。"""
        return self.b * (self.rho + (k - self.m) / np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def d2w(self, k):
        """w''(k)。"""
        return self.b * self.sigma ** 2 / np.power((k - self.m) ** 2 + self.sigma ** 2, 1.5)

    def iv(self, k, maturity: float):
        """隐含波动率 σ(k) = sqrt(w(k)/T)。"""
        w = np.maximum(np.asarray(self.w(k), dtype="float64"), 0.0)
        return np.sqrt(w / float(maturity))

    def g(self, k):
        """Durrleman 条件函数：``g(k) ≥ 0`` 等价于蝶式无套利（密度非负）。"""
        k = np.asarray(k, dtype="float64")
        w = np.asarray(self.w(k), dtype="float64")
        dw = np.asarray(self.dw(k), dtype="float64")
        d2w = np.asarray(self.d2w(k), dtype="float64")
        safe = np.maximum(w, 1e-14)
        return ((1.0 - k * dw / (2.0 * safe)) ** 2
                - (dw / 4.0) * (1.0 / safe + 0.25)
                + d2w / 2.0)

    def arbitrage_metrics(self, k_grid: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """返回 ``w_min`` / ``g_min`` 与是否无套利（在给定网格上）。"""
        if k_grid is None:
            k_grid = np.linspace(-3.0, 3.0, 601)
        k_grid = np.asarray(k_grid, dtype="float64")
        w = np.asarray(self.w(k_grid), dtype="float64")
        g = np.asarray(self.g(k_grid), dtype="float64")
        return {
            "w_min": float(w.min()),
            "g_min": float(g.min()),
            "butterfly_free": bool(g.min() >= -1e-10),
            "variance_positive": bool(w.min() >= -1e-12),
        }

    def to_dict(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m, "sigma": self.sigma}

    # ------------------------------------------------------------------ 拟合
    @classmethod
    def fit(
        cls,
        k: Sequence[float],
        w: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        *,
        refine: bool = True,
    ) -> "SVISlice":
        """加权最小二乘拟合（网格初始化 + Nelder-Mead 精修）。"""
        k_arr, w_arr = _as_arrays(k, w)
        wts = np.ones_like(k_arr) if weights is None else np.asarray(weights, dtype="float64").ravel()
        wts = np.broadcast_to(wts, k_arr.shape).copy()

        best = cls._grid_search(k_arr, w_arr, wts)
        if not refine:
            return best
        return cls._refine(best, k_arr, w_arr, wts)

    @staticmethod
    def _fit_ab(k: np.ndarray, w: np.ndarray, wts: np.ndarray, rho: float,
                m: float, sigma: float):
        """固定 (ρ,m,σ) 时 (a,b) 是线性参数 -> 最小二乘。"""
        shape = rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2)
        design = np.column_stack([np.ones_like(k), shape])
        sqrt_w = np.sqrt(wts)
        coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], w * sqrt_w, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        return a, max(b, 1e-8)

    @classmethod
    def _grid_search(cls, k: np.ndarray, w: np.ndarray, wts: np.ndarray) -> "SVISlice":
        span = float(k.max() - k.min())
        m_grid = np.linspace(k.min(), k.max(), 5) if span > 0 else np.array([0.0])
        sigma_grid = np.array([0.03, 0.08, 0.15, 0.3]) * max(1.0, span)
        rho_grid = np.array([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])

        best: Optional[SVISlice] = None
        best_sse = np.inf
        for rho in rho_grid:
            for m in m_grid:
                for sigma in sigma_grid:
                    a, b = cls._fit_ab(k, w, wts, float(rho), float(m), float(sigma))
                    candidate = cls(a, b, float(rho), float(m), float(sigma))
                    sse = weighted_sse(candidate.w(k), w, wts)
                    if sse < best_sse:
                        best_sse, best = sse, candidate
        assert best is not None
        return best

    @classmethod
    def _refine(cls, initial: "SVISlice", k: np.ndarray, w: np.ndarray,
                wts: np.ndarray) -> "SVISlice":
        penalty_grid = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 101)

        def unpack(x: np.ndarray) -> "SVISlice":
            return cls(a=float(x[0]), b=float(np.exp(x[1])),
                       rho=float(np.tanh(x[2])), m=float(x[3]),
                       sigma=float(np.exp(x[4])))

        def objective(x: np.ndarray) -> float:
            cand = unpack(x)
            sse = weighted_sse(cand.w(k), w, wts)
            w_grid = np.asarray(cand.w(penalty_grid), dtype="float64")
            sse += 1e3 * float(np.sum(np.minimum(w_grid, 0.0) ** 2))
            if abs(cand.rho) >= 0.999 or cand.b > 50 or cand.sigma > 10:
                sse += 1e3
            return sse

        x0 = np.array([
            initial.a,
            np.log(max(initial.b, 1e-8)),
            np.arctanh(max(min(initial.rho, 0.999), -0.999)),
            initial.m,
            np.log(max(initial.sigma, 1e-8)),
        ])
        x_best, _ = nelder_mead(objective, x0, step=0.02, max_iter=3000, tol=1e-14)
        refined = unpack(x_best)
        # 精修后若反而更差（数值异常），回退到网格结果
        if weighted_sse(refined.w(k), w, wts) > weighted_sse(initial.w(k), w, wts):
            return initial
        return refined


def fit_svi_slice(k: Sequence[float], total_variance: Sequence[float],
                  **kwargs: Any) -> SVISlice:
    """便捷函数：按总方差拟合一条 SVI。"""
    return SVISlice.fit(k, total_variance, **kwargs)


__all__ = ["SVISlice", "fit_svi_slice"]