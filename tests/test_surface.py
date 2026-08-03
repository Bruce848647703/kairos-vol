"""曲面层测试：SVI / SSVI / 插值 / 无套利。"""
from __future__ import annotations

import numpy as np
import pytest

from kairos_vol.arbitrage import (
    butterfly_report,
    calendar_report,
    risk_neutral_density,
    surface_report,
)
from kairos_vol.data import synthetic_ssvi_quotes
from kairos_vol.ssvi import SSVISurface
from kairos_vol.surface import Quote, VolSurface
from kairos_vol.svi import SVISlice

TRUE_SVI = {"a": 0.015, "b": 0.12, "rho": -0.35, "m": 0.05, "sigma": 0.25}


def test_svi_recovers_synthetic_slice():
    true = SVISlice(**TRUE_SVI)
    k = np.linspace(-1.0, 1.0, 41)
    w = np.asarray(true.w(k))
    fit = SVISlice.fit(k, w)
    assert float(np.max(np.abs(np.asarray(fit.w(k)) - w))) < 1e-4
    assert fit.arbitrage_metrics()["butterfly_free"]


def test_svi_derivatives_match_finite_difference():
    sl = SVISlice(**TRUE_SVI)
    k = 0.1
    h = 1e-5
    dw = (float(sl.w(k + h)) - float(sl.w(k - h))) / (2 * h)
    d2w = (float(sl.w(k + h)) - 2 * float(sl.w(k)) + float(sl.w(k - h))) / (h * h)
    assert dw == pytest.approx(float(sl.dw(k)), rel=1e-6)
    assert d2w == pytest.approx(float(sl.d2w(k)), rel=1e-4)


def test_svi_fit_requires_enough_points():
    with pytest.raises(ValueError):
        SVISlice.fit([0.0, 0.1], [0.01, 0.02])


def test_butterfly_detector_flags_bad_slices():
    flagged = 0
    grid = np.linspace(-1.0, 1.0, 201)
    for b in (0.5, 1.0, 2.0, 4.0):
        for sigma in (0.005, 0.01, 0.02, 0.05):
            sl = SVISlice(a=0.0, b=b, rho=0.9, m=0.0, sigma=sigma)
            if butterfly_report(sl, grid)["n_violations"] > 0:
                flagged += 1
    assert flagged > 0


def test_ssvi_recovers_parameters():
    true = SSVISurface(rho=-0.6, eta=0.8, gamma=0.3)
    ks = np.linspace(-0.5, 0.5, 21)
    slices = []
    for t in (0.1, 0.25, 0.5, 1.0, 2.0):
        theta = 0.04 * t
        slices.append((theta, ks, np.asarray(true.w(ks, theta))))
    fit = SSVISurface.fit(slices)
    assert fit.rho == pytest.approx(-0.6, abs=1e-3)
    assert fit.eta == pytest.approx(0.8, abs=1e-3)
    assert fit.gamma == pytest.approx(0.3, abs=1e-3)
    assert fit.no_arb_ok()


def test_ssvi_validation():
    with pytest.raises(ValueError):
        SSVISurface(rho=1.5, eta=1.0, gamma=0.2)
    with pytest.raises(ValueError):
        SSVISurface(rho=-0.5, eta=1.0, gamma=0.6)


def test_surface_from_quotes_recovers_atm():
    quotes = synthetic_ssvi_quotes()
    surface = VolSurface.fit(quotes)
    assert surface.n_slices == 5
    for t in surface.maturities:
        expected = float(np.interp(t, (0.08, 0.25, 0.5, 1.0, 2.0),
                                   (0.21, 0.20, 0.185, 0.175, 0.17)))
        assert surface.atm_vol(t) == pytest.approx(expected, abs=2e-3)


def test_surface_total_variance_monotone_in_maturity():
    surface = VolSurface.fit(synthetic_ssvi_quotes())
    for k in (-0.4, 0.0, 0.4):
        ws = [float(np.asarray(surface.total_variance(k, t))) for t in surface.maturities]
        assert all(b >= a - 1e-10 for a, b in zip(ws, ws[1:]))


def test_surface_interpolates_between_pillars():
    surface = VolSurface.fit(synthetic_ssvi_quotes())
    mid = 0.5 * (surface.maturities[0] + surface.maturities[1])
    w0 = float(np.asarray(surface.total_variance(0.0, surface.maturities[0])))
    w1 = float(np.asarray(surface.total_variance(0.0, surface.maturities[1])))
    wm = float(np.asarray(surface.total_variance(0.0, mid)))
    assert min(w0, w1) - 1e-12 <= wm <= max(w0, w1) + 1e-12


def test_calendar_violation_detected():
    low = SVISlice(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    high = SVISlice(a=0.03, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    surface = VolSurface([0.5, 1.0], [low, high], [100.0, 100.0])
    report = calendar_report(surface)
    assert report["calendar_free"] is False
    assert report["violations"]


def test_surface_report_is_arbitrage_free_for_synthetic():
    surface = VolSurface.fit(synthetic_ssvi_quotes())
    report = surface_report(surface)
    assert report["arbitrage_free"] is True
    assert report["calendar"]["calendar_free"] is True


def test_risk_neutral_density_integrates_to_one():
    surface = VolSurface.fit(synthetic_ssvi_quotes(noise=0.0))
    density = risk_neutral_density(surface, 1.0, n=301, k_range=(-3.0, 3.0))
    assert 0.95 <= density["integral"] <= 1.05
    assert density["min_density"] > -1e-3


def test_quote_validation():
    with pytest.raises(ValueError):
        Quote(0.0, 100, 0.2, 100)
    with pytest.raises(ValueError):
        Quote(1.0, 100, -0.1, 100)
    assert Quote(1.0, 100, 0.2, 100).total_variance == pytest.approx(0.04)