"""示例 3：PnL 归因 —— 一天里赚的钱是 Delta、Gamma、Vega 还是 Theta？

运行：python examples/03_pnl_attribution.py
"""
from __future__ import annotations

from kairos_vol.data import synthetic_ssvi_quotes
from kairos_vol.hedging import explain_pnl
from kairos_vol.surface import VolSurface


def main() -> None:
    # 今天的曲面
    surface = VolSurface.fit(synthetic_ssvi_quotes())

    # 明天：标的涨 1%，曲面整体上移 0.5 个波动率点（用同形状曲面近似平移）
    shifted_quotes = synthetic_ssvi_quotes(atm_vol=(0.215, 0.205, 0.19, 0.18, 0.175))
    tomorrow = VolSurface.fit(shifted_quotes)

    for strike in (90.0, 100.0, 110.0):
        row = explain_pnl(100.0, 101.0, surface, tomorrow, strike, 0.5, 0.497, kind="call")
        print(f"K={strike:>5.0f}  实际 {row['actual']:+.4f}  =  "
              f"Δ {row['delta']:+.4f}  Γ {row['gamma']:+.4f}  "
              f"ν {row['vega']:+.4f}  Θ {row['theta']:+.4f}  "
              f"| 未解释 {row['unexplained']:+.5f}")

    print("\n（未解释项是二阶/交叉项与曲面形状变化，健康值应远小于实际盈亏）")


if __name__ == "__main__":
    main()