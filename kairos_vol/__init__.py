"""Kairos Vol —— 自研期权波动率研究库。

覆盖从「报价」到「波动率交易」的完整链路：

- **定价**：Black-76 / Black-Scholes + 解析希腊字母（仅 numpy + math）
- **隐含波动率**：Newton + 二分保护的稳健反解，支持向量化与坏点标记
- **曲面**：SVI 逐期限校准 → 总方差沿期限插值；SSVI 全局 3 参数拟合
- **无套利**：蝶式 ``g(k)≥0``、日历 ``w`` 单调、Breeden-Litzenberger 密度
- **局部波动率**：Dupire 公式（总方差形式）
- **方差互换**：对数合约复制、已实现方差、vol carry、期限结构
- **PnL 归因**：把期权盈亏拆成 Delta / Gamma / Vega / Theta

快速开始::

    pip install -e .
    python -m kairos_vol            # 合成曲面 -> 校准 -> 体检 -> 方差互换
    python -m kairos_vol price --spot 100 --strike 100 --vol 0.2
"""
from __future__ import annotations

from .arbitrage import (
    butterfly_report,
    calendar_report,
    check_quotes,
    risk_neutral_density,
    surface_report,
)
from .core import (
    black76,
    black_scholes,
    implied_forward,
    intrinsic,
    norm_cdf,
    norm_pdf,
    put_call_parity_gap,
)
from .data import (
    load_quotes_csv,
    save_quotes_csv,
    surface_from_quotes,
    synthetic_ssvi_quotes,
)
from .hedging import explain_pnl, explain_table
from .implied import implied_vol, implied_vol_grid, total_variance
from .localvol import dupire_local_variance, dupire_local_vol, local_vol_grid
from .ssvi import SSVISurface
from .surface import Quote, VolSurface, quotes_from_grid
from .svi import SVISlice, fit_svi_slice
from .variance import (
    fair_variance_strike,
    fair_vol_strike,
    realized_variance,
    term_structure,
    variance_swap_pnl,
    vol_carry,
)

__version__ = "0.1.1"

__all__ = [
    "Quote",
    "SSVISurface",
    "SVISlice",
    "VolSurface",
    "__version__",
    "black76",
    "black_scholes",
    "butterfly_report",
    "calendar_report",
    "check_quotes",
    "dupire_local_variance",
    "dupire_local_vol",
    "explain_pnl",
    "explain_table",
    "fair_variance_strike",
    "fair_vol_strike",
    "fit_svi_slice",
    "implied_forward",
    "implied_vol",
    "implied_vol_grid",
    "intrinsic",
    "load_quotes_csv",
    "local_vol_grid",
    "norm_cdf",
    "norm_pdf",
    "put_call_parity_gap",
    "quotes_from_grid",
    "realized_variance",
    "risk_neutral_density",
    "save_quotes_csv",
    "surface_from_quotes",
    "surface_report",
    "synthetic_ssvi_quotes",
    "term_structure",
    "total_variance",
    "variance_swap_pnl",
    "vol_carry",
]