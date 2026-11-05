"""期权的 PnL 归因：把价格变化拆成 Delta / Gamma / Vega / Theta。

这是波动率交易台的日常工具：每天收盘后解释「今天赚的钱是哪来的」，
剩下的 ``unexplained`` 就是模型/曲面/执行误差。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from .core import black_scholes
from .surface import VolSurface


def explain_pnl(
    spot0: float,
    spot1: float,
    surface0: VolSurface,
    surface1: VolSurface,
    strike: float,
    t0: float,
    t1: float,
    kind: str = "call",
    rate: float = 0.0,
    dividend: float = 0.0,
) -> Dict[str, Any]:
    """解释一张期权从 ``(spot0, surface0, t0)`` 到 ``(spot1, surface1, t1)`` 的 PnL。"""
    forward0 = float(spot0) * math.exp((float(rate) - float(dividend)) * float(t0))
    iv0 = float(surface0.iv_at(strike, t0, forward=forward0))
    forward1 = float(spot1) * math.exp((float(rate) - float(dividend)) * float(t1))
    iv1 = float(surface1.iv_at(strike, t1, forward=forward1))

    p0 = black_scholes(spot0, strike, t0, rate, iv0, dividend, kind)
    p1 = black_scholes(spot1, strike, t1, rate, iv1, dividend, kind)

    ds = float(spot1) - float(spot0)

    delta_pnl = p0["delta"] * ds
    gamma_pnl = 0.5 * p0["gamma"] * ds * ds
    vega_pnl = p0["vega"] * (iv1 - iv0)
    theta_pnl = p0["theta"] * (float(t0) - float(t1))   # theta 为日历口径
    explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    actual = p1["price"] - p0["price"]
    unexplained = actual - explained
    return {
        "actual": actual,
        "delta": delta_pnl,
        "gamma": gamma_pnl,
        "vega": vega_pnl,
        "theta": theta_pnl,
        "explained": explained,
        "unexplained": unexplained,
        "unexplained_ratio": (unexplained / actual) if abs(actual) > 1e-14 else 0.0,
        "iv0": iv0,
        "iv1": iv1,
        "price0": p0["price"],
        "price1": p1["price"],
        "spot_move": ds,
        "vol_move": iv1 - iv0,
    }


def explain_table(
    spot0: float,
    spot1: float,
    surface0: VolSurface,
    surface1: VolSurface,
    strikes: Sequence[float],
    t0: float,
    t1: float,
    kind: str = "call",
    rate: float = 0.0,
    dividend: float = 0.0,
) -> Dict[str, Any]:
    """对一组行权价做批量归因，返回逐行结果与合计。"""
    rows = [explain_pnl(spot0, spot1, surface0, surface1, k, t0, t1, kind,
                        rate, dividend) for k in strikes]
    total_actual = sum(r["actual"] for r in rows)
    total_unexplained = sum(r["unexplained"] for r in rows)
    return {
        "rows": rows,
        "total_actual": total_actual,
        "total_unexplained": total_unexplained,
        "unexplained_ratio": (total_unexplained / total_actual)
        if abs(total_actual) > 1e-14 else 0.0,
    }


__all__ = ["explain_pnl", "explain_table"]