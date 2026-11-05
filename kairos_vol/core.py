"""定价核心：Black-76（远期）与 Black-Scholes（现货），含解析希腊字母。

只依赖 ``numpy`` 与标准库 ``math``；正态分布用 ``math.erf``，不需要 scipy。
所有价格都是「每 1 单位标的/合约」的价格，乘数由调用方处理。
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
KINDS = ("call", "put")


def norm_cdf(x: float) -> float:
    """标准正态分布函数 Φ(x)。"""
    return 0.5 * (1.0 + math.erf(float(x) / _SQRT_2))


def norm_pdf(x: float) -> float:
    """标准正态密度 φ(x)。"""
    return math.exp(-0.5 * float(x) ** 2) / _SQRT_2PI


def norm_cdf_array(x: np.ndarray) -> np.ndarray:
    """向量化 Φ(x)（内部仍是 math.erf，避免引入 scipy）。"""
    arr = np.asarray(x, dtype="float64")
    return np.array([norm_cdf(v) for v in arr.ravel()], dtype="float64").reshape(arr.shape)


def _check_kind(kind: str) -> str:
    kind = str(kind).lower()
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS}，收到 {kind!r}")
    return kind


def _check_inputs(forward: float, strike: float, maturity: float, vol: float) -> None:
    if not forward > 0:
        raise ValueError(f"forward 必须为正，收到 {forward!r}")
    if not strike > 0:
        raise ValueError(f"strike 必须为正，收到 {strike!r}")
    if not maturity >= 0:
        raise ValueError(f"maturity 不能为负，收到 {maturity!r}")
    if not vol >= 0:
        raise ValueError(f"vol 不能为负，收到 {vol!r}")


def intrinsic(forward: float, strike: float, kind: str = "call", discount: float = 1.0) -> float:
    """贴现后的内在价值。"""
    kind = _check_kind(kind)
    value = max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0)
    return value * float(discount)


def black76(
    forward: float,
    strike: float,
    maturity: float,
    vol: float,
    discount: float = 1.0,
    kind: str = "call",
) -> Dict[str, Any]:
    """Black-76：以远期价格为标的、``discount = e^{-rT}`` 为贴现因子。

    返回值：``price / delta / gamma / vega / theta / d1 / d2 / intrinsic``。
    其中 ``delta`` 是对**远期**的敏感度，``vega`` 是波动率变动 1.00 的价格变化，
    ``theta`` 是对到期时间 T 的敏感度（远期保持不变）。
    """
    kind = _check_kind(kind)
    _check_inputs(forward, strike, maturity, vol)
    forward, strike, vol = float(forward), float(strike), float(vol)
    discount = float(discount)
    sign = 1.0 if kind == "call" else -1.0
    intr = intrinsic(forward, strike, kind, discount)

    if maturity == 0.0:
        return {
            "price": intr, "delta": discount * (1.0 if sign * (forward - strike) > 0 else 0.0),
            "gamma": 0.0, "vega": 0.0, "theta": 0.0, "d1": 0.0, "d2": 0.0,
            "intrinsic": intr, "kind": kind, "forward": forward, "discount": discount,
        }
    if vol == 0.0:
        forward_at_expiry = forward
        payoff = max(sign * (forward_at_expiry - strike), 0.0)
        delta = discount * (1.0 if sign * (forward - strike) > 0 else 0.0)
        return {
            "price": payoff * discount, "delta": delta, "gamma": 0.0, "vega": 0.0,
            "theta": 0.0, "d1": 0.0, "d2": 0.0, "intrinsic": intr, "kind": kind,
            "forward": forward, "discount": discount,
        }

    sqrt_t = math.sqrt(maturity)
    vol_sqrt_t = vol * sqrt_t
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    pdf_d1 = norm_pdf(d1)

    if kind == "call":
        price = discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
        delta = discount * norm_cdf(d1)
    else:
        price = discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
        delta = -discount * norm_cdf(-d1)

    return {
        "price": max(price, 0.0),
        "delta": delta,
        "gamma": discount * pdf_d1 / (forward * vol_sqrt_t),
        "vega": discount * forward * pdf_d1 * sqrt_t,
        "theta": -discount * forward * pdf_d1 * vol / (2.0 * sqrt_t),
        "d1": d1, "d2": d2, "intrinsic": intr, "kind": kind,
        "forward": forward, "discount": discount,
    }


def black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend: float = 0.0,
    kind: str = "call",
) -> Dict[str, Any]:
    """Black-Scholes：现货 + 连续分红率，返回对**现货**的希腊字母。

    ``theta`` 用中心差分计算（对 T，保持 S 不变），避免手推公式的符号错误。
    """
    kind = _check_kind(kind)
    if not spot > 0:
        raise ValueError(f"spot 必须为正，收到 {spot!r}")
    spot, strike = float(spot), float(strike)
    rate, dividend, vol = float(rate), float(dividend), float(vol)
    if maturity < 0:
        raise ValueError(f"maturity 不能为负，收到 {maturity!r}")
    if vol < 0:
        raise ValueError(f"vol 不能为负，收到 {vol!r}")

    discount = math.exp(-rate * maturity)
    forward = spot * math.exp((rate - dividend) * maturity)
    res = black76(forward, strike, maturity, vol, discount, kind)
    res["spot"] = spot
    res["forward"] = forward
    res["discount"] = discount
    # Black-76 的希腊字母是对远期 F 的；换算到现货：
    #   delta_S = delta_F · e^{(r-q)T}，gamma_S = gamma_F · e^{2(r-q)T}，vega 口径相同
    growth = math.exp((rate - dividend) * maturity)
    res["delta"] = res["delta"] * growth
    res["gamma"] = res["gamma"] * growth * growth
    res["vega"] = res["vega"]

    if maturity > 0:
        h = min(1e-5, maturity * 1e-3)
        up = black76(spot * math.exp((rate - dividend) * (maturity + h)), strike,
                     maturity + h, vol, math.exp(-rate * (maturity + h)), kind)["price"]
        dn = black76(spot * math.exp((rate - dividend) * (maturity - h)), strike,
                     max(maturity - h, 1e-9), vol, math.exp(-rate * (max(maturity - h, 1e-9))),
                     kind)["price"]
        res["theta"] = (dn - up) / (2.0 * h)
    return res


def put_call_parity_gap(call_price: float, put_price: float, forward: float,
                        strike: float, discount: float = 1.0) -> float:
    """``C - P - (F - K)·df``；理论上为 0，用于数据质检。"""
    return float(call_price) - float(put_price) - (float(forward) - float(strike)) * float(discount)


def implied_forward(call_price: float, put_price: float, strike: float,
                    discount: float = 1.0) -> float:
    """由同一行权价的看涨/看跌价格反解远期（C - P = df·(F - K)）。"""
    return float(strike) + (float(call_price) - float(put_price)) / float(discount)


__all__ = [
    "KINDS",
    "black76",
    "black_scholes",
    "implied_forward",
    "intrinsic",
    "norm_cdf",
    "norm_cdf_array",
    "norm_pdf",
    "put_call_parity_gap",
]