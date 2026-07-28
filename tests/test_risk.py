"""局部波动率、方差互换与 PnL 归因测试。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kairos_vol.data import synthetic_ssvi_quotes
from kairos_vol.hedging import explain_pnl, explain_table
from kairos_vol.localvol import dupire_local_variance, dupire_local_vol, local_vol_grid
from kairos_vol.surface import VolSurface
from kairos_vol.svi import SVISlice
from kairos_vol.variance import (
    fair_variance_strike,
    fair_vol_strike,
    realized_variance,
    term_structure,
    variance_swap_pnl,
    vol_carry,
)


def flat_surface(sigma: float = 0.2, maturities=(0.25, 1.0, 2.0),
                 forward: float = 100.0) -> VolSurface:
    slices = [SVISlice(a=sigma ** 2 * t, b=0.0, rho=0.0, m=0.0, sigma=0.1)
              for t in maturities]
    return VolSurface(list(maturities), slices, [forward] * len(maturities))


# --------------------------------------------------------------------------- #
# Dupire 局部波动率
# --------------------------------------------------------------------------- #
def test_dupire_flat_surface_gives_constant_vol():
    surface = flat_surface(0.2)
    for t in (0.25, 1.0, 2.0):
        assert dupire_local_vol(surface, 0.0, t) == pytest.approx(0.2, rel=1e-6)
        assert dupire_local_vol(surface, 0.3, t) == pytest.approx(0.2, rel=1e-6)
    assert dupire_local_variance(surface, 0.0, 1.0) == pytest.approx(0.04, rel=1e-6)


def test_dupire_on_synthetic_surface_is_finite_and_positive():
    surface = VolSurface.fit(synthetic_ssvi_quotes())
    grid = local_vol_grid(surface, k_range=(-0.3, 0.3), n_k=7)
    values = np.asarray(grid["local_vol"], dtype="float64")
    assert values.shape == (len(surface.maturities), 7)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)


# --------------------------------------------------------------------------- #
# 方差互换
# --------------------------------------------------------------------------- #
def test_fair_variance_equals_sigma_squared_for_flat_surface():
    surface = flat_surface(0.2)
    var = fair_variance_strike(surface, 1.0)
    assert var == pytest.approx(0.04, abs=2e-3)
    assert fair_vol_strike(surface, 1.0) == pytest.approx(0.2, abs=5e-3)


def test_vol_carry_positive_for_downward_skew():
    surface = VolSurface.fit(synthetic_ssvi_quotes())
    row = vol_carry(surface, 1.0)
    assert row["fair_variance"] > row["atm_variance"]
    assert row["carry_vol"] > 0
    assert len(term_structure(surface)) == surface.n_slices


def test_realized_variance_known_series():
    returns = np.full(252, 0.01)
    prices = 100.0 * np.exp(np.cumsum(returns))
    var = realized_variance(prices)
    assert var == pytest.approx(1e-4 * 252, rel=1e-3)


def test_variance_swap_pnl():
    assert variance_swap_pnl(0.05, 0.04, 1000.0) == pytest.approx(10.0)
    assert variance_swap_pnl(0.03, 0.04, 1000.0) == pytest.approx(-10.0)
    with pytest.raises(ValueError):
        realized_variance([100.0])


# --------------------------------------------------------------------------- #
# PnL 归因
# --------------------------------------------------------------------------- #
def test_pnl_attribution_residual_small_on_flat_surface():
    surface = flat_surface(0.2)
    row = explain_pnl(100.0, 100.5, surface, surface, 100.0, 0.5, 0.5)
    assert row["actual"] > 0
    assert abs(row["unexplained"]) < 1e-3 * abs(row["actual"])
    assert row["vega"] == pytest.approx(0.0, abs=1e-12)


def test_pnl_attribution_captures_vega():
    base = flat_surface(0.20)
    shifted = flat_surface(0.21)
    row = explain_pnl(100.0, 100.0, base, shifted, 100.0, 0.5, 0.5)
    assert row["vol_move"] == pytest.approx(0.01, abs=1e-12)
    assert row["actual"] == pytest.approx(row["vega"], rel=0.01)


def test_pnl_attribution_table_totals():
    surface = flat_surface(0.2)
    table = explain_table(100.0, 101.0, surface, surface, [95.0, 100.0, 105.0],
                          0.5, 0.5)
    assert len(table["rows"]) == 3
    assert table["total_actual"] == pytest.approx(sum(r["actual"] for r in table["rows"]))
    assert abs(table["unexplained_ratio"]) < 1e-2


def test_pnl_attribution_with_theta():
    surface = flat_surface(0.2)
    row = explain_pnl(100.0, 100.0, surface, surface, 100.0, 0.50, 0.48)
    assert row["theta"] < 0                      # 多头期权的时间价值衰减
    assert row["actual"] == pytest.approx(row["theta"], rel=0.05)