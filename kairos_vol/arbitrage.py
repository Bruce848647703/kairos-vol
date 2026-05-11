"""无套利检验：蝶式（密度非负）、日历（总方差单调）、风险中性密度。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .core import black76

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
from .svi import SVISlice
from .surface import VolSurface


def default_k_grid(k_min: float = -1.5, k_max: float = 1.5,
                   n: int = 241) -> np.ndarray:
    return np.linspace(k_min, k_max, int(n))


def butterfly_report(slice: SVISlice,
                     k_grid: Optional[Sequence[float]] = None,
                     tol: float = 1e-10) -> Dict[str, Any]:
    """单条微笑的蝶式检查：``g(k) ≥ 0`` 等价于风险中性密度非负。"""
    ks = default_k_grid() if k_grid is None else np.asarray(k_grid, dtype="float64")
    g = np.asarray(slice.g(ks), dtype="float64")
    w = np.asarray(slice.w(ks), dtype="float64")
    worst = int(np.argmin(g))
    return {
        "butterfly_free": bool(g.min() >= -tol),
        "g_min": float(g.min()),
        "g_argmin_k": float(ks[worst]),
        "w_min": float(w.min()),
        "variance_positive": bool(w.min() >= -tol),
        "n_violations": int(np.sum(g < -tol)),
    }


def calendar_report(surface: VolSurface,
                    k_grid: Optional[Sequence[float]] = None,
                    tol: float = 1e-10) -> Dict[str, Any]:
    """相邻期限的总方差必须随期限不减：``w(k, T₂) ≥ w(k, T₁)``。"""
    ks = default_k_grid() if k_grid is None else np.asarray(k_grid, dtype="float64")
    violations: List[Dict[str, float]] = []
    min_delta = np.inf
    for i in range(1, len(surface.maturities)):
        t0, t1 = surface.maturities[i - 1], surface.maturities[i]
        w0 = np.asarray(surface.slices[i - 1].w(ks), dtype="float64")
        w1 = np.asarray(surface.slices[i].w(ks), dtype="float64")
        delta = w1 - w0
        min_delta = min(min_delta, float(delta.min()))
        if delta.min() < -tol:
            violations.append({"t0": t0, "t1": t1, "min_delta": float(delta.min())})
    return {
        "calendar_free": not violations,
        "min_delta_w": float(min_delta) if np.isfinite(min_delta) else 0.0,
        "violations": violations,
    }


def surface_report(surface: VolSurface,
                   k_grid: Optional[Sequence[float]] = None,
                   tol: float = 1e-10) -> Dict[str, Any]:
    """整条曲面的无套利体检报告。"""
    ks = default_k_grid() if k_grid is None else np.asarray(k_grid, dtype="float64")
    butterflies = {
        f"T={t:g}": butterfly_report(s, ks, tol)
        for t, s in zip(surface.maturities, surface.slices)
    }
    calendar = calendar_report(surface, ks, tol)
    return {
        "arbitrage_free": bool(calendar["calendar_free"]
                               and all(b["butterfly_free"] for b in butterflies.values())),
        "butterfly": butterflies,
        "calendar": calendar,
        "k_grid": {"min": float(ks.min()), "max": float(ks.max()), "n": int(ks.size)},
    }


def risk_neutral_density(surface: VolSurface, maturity: float,
                         n: int = 241, k_range=(-2.0, 2.0)) -> Dict[str, Any]:
    """Breeden-Litzenberger：``f(K) = ∂²C/∂K²``（以远期价、贴现因子=1 表示）。"""
    forward = surface.forward(maturity)
    ks = np.linspace(k_range[0], k_range[1], int(n))
    strikes = forward * np.exp(ks)
    ivs = np.asarray(surface.iv(ks, maturity), dtype="float64")
    calls = np.array([
        black76(forward, float(k), maturity, float(iv), 1.0, "call")["price"]
        for k, iv in zip(strikes, ivs)
    ], dtype="float64")
    density = np.gradient(np.gradient(calls, strikes), strikes)
    integral = float(_trapz(density, strikes))
    return {
        "maturity": float(maturity),
        "forward": forward,
        "strikes": strikes.tolist(),
        "density": density.tolist(),
        "integral": integral,
        "negative_share": float(np.mean(density < -1e-10)),
        "min_density": float(density.min()),
        "max_density": float(density.max()),
    }


def check_quotes(quotes: Sequence[Any], tol: float = 1e-8) -> Dict[str, Any]:
    """原始报价的快速质检（不拟曲面）。"""
    issues: List[str] = []
    total_variances = []
    for quote in quotes:
        if quote.iv <= 0:
            issues.append(f"T={quote.maturity:g} K={quote.strike:g}: 非正波动率")
        total_variances.append(quote.total_variance)
    total_variances = np.asarray(total_variances, dtype="float64")
    return {
        "n_quotes": len(quotes),
        "min_total_variance": float(total_variances.min()) if total_variances.size else 0.0,
        "negative_total_variance": int(np.sum(total_variances <= 0)),
        "issues": issues,
        "ok": bool(not issues),
    }


__all__ = [
    "butterfly_report",
    "calendar_report",
    "check_quotes",
    "default_k_grid",
    "risk_neutral_density",
    "surface_report",
]