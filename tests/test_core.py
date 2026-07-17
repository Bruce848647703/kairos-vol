"""定价与隐含波动率测试。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kairos_vol.core import (
    black76,
    black_scholes,
    implied_forward,
    intrinsic,
    norm_cdf,
    put_call_parity_gap,
)
from kairos_vol.implied import implied_vol, implied_vol_grid


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975002, abs=1e-6)
    assert norm_cdf(-1.96) == pytest.approx(0.024998, abs=1e-6)


def test_black76_put_call_parity():
    f, k, t, vol, df = 100.0, 105.0, 0.75, 0.25, math.exp(-0.03 * 0.75)
    call = black76(f, k, t, vol, df, "call")["price"]
    put = black76(f, k, t, vol, df, "put")["price"]
    assert call - put == pytest.approx(df * (f - k), rel=1e-12)


def test_black_scholes_matches_black76():
    s, k, t, r, q, vol = 100.0, 100.0, 1.0, 0.03, 0.01, 0.25
    f = s * math.exp((r - q) * t)
    df = math.exp(-r * t)
    direct = black_scholes(s, k, t, r, vol, q, "call")
    fwd = black76(f, k, t, vol, df, "call")
    assert direct["price"] == pytest.approx(fwd["price"], rel=1e-12)
    growth = math.exp((r - q) * t)
    assert direct["delta"] == pytest.approx(fwd["delta"] * growth, rel=1e-12)
    assert direct["gamma"] == pytest.approx(fwd["gamma"] * growth * growth, rel=1e-12)
    assert direct["vega"] == pytest.approx(fwd["vega"], rel=1e-12)


def test_expired_option_is_intrinsic():
    assert black_scholes(120, 100, 0.0, 0.02, 0.2, kind="call")["price"] == pytest.approx(20.0)
    assert black_scholes(80, 100, 0.0, 0.02, 0.2, kind="put")["price"] == pytest.approx(20.0)
    assert intrinsic(90, 100, "call") == 0.0


def test_greeks_match_finite_difference():
    s, k, t, r, vol = 100.0, 100.0, 0.75, 0.02, 0.25
    base = black_scholes(s, k, t, r, vol)
    h = 1e-4

    up = black_scholes(s + h, k, t, r, vol)["price"]
    dn = black_scholes(s - h, k, t, r, vol)["price"]
    assert (up - dn) / (2 * h) == pytest.approx(base["delta"], rel=1e-6)
    assert (up - 2 * base["price"] + dn) / (h * h) == pytest.approx(base["gamma"], rel=1e-4)

    vup = black_scholes(s, k, t, r, vol + h)["price"]
    vdn = black_scholes(s, k, t, r, vol - h)["price"]
    assert (vup - vdn) / (2 * h) == pytest.approx(base["vega"], rel=1e-6)
    assert base["theta"] < 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        black76(-1, 100, 1, 0.2)
    with pytest.raises(ValueError):
        black76(100, 100, 1, 0.2, kind="straddle")
    with pytest.raises(ValueError):
        black_scholes(0, 100, 1, 0.02, 0.2)


def test_put_call_parity_helpers():
    df = 0.98
    f = 101.0
    k = 100.0
    call = black76(f, k, 0.5, 0.2, df, "call")["price"]
    put = black76(f, k, 0.5, 0.2, df, "put")["price"]
    assert put_call_parity_gap(call, put, f, k, df) == pytest.approx(0.0, abs=1e-12)
    assert implied_forward(call, put, k, df) == pytest.approx(f, rel=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.8])
@pytest.mark.parametrize("moneyness", [80.0, 100.0, 120.0])
def test_implied_vol_roundtrip(kind, vol, moneyness):
    f, t, df = 100.0, 1.0, 0.97
    price = black76(f, moneyness, t, vol, df, kind)["price"]
    solved = implied_vol(price, f, moneyness, t, df, kind)
    assert solved == pytest.approx(vol, rel=1e-6)


def test_implied_vol_rejects_arbitrage_violation():
    with pytest.raises(ValueError, match="无套利区间"):
        implied_vol(1000.0, 100.0, 100.0, 1.0, 1.0, "call")
    with pytest.raises(ValueError, match="无套利区间"):
        implied_vol(0.001, 100.0, 50.0, 1.0, 1.0, "call")


def test_implied_vol_grid_marks_bad_points_as_nan():
    prices = [10.0, 1000.0, 10.0]
    vols = implied_vol_grid(prices, [100.0] * 3, [100.0] * 3, [1.0] * 3)
    assert np.isfinite(vols[0]) and np.isfinite(vols[2])
    assert np.isnan(vols[1])