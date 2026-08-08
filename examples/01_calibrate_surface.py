"""示例 1：合成报价 -> SVI 校准 -> 无套利体检 -> 与真参数对比。

运行：python examples/01_calibrate_surface.py
"""
from __future__ import annotations

import numpy as np

from kairos_vol.arbitrage import surface_report
from kairos_vol.data import synthetic_ssvi_quotes
from kairos_vol.ssvi import SSVISurface
from kairos_vol.surface import VolSurface


def main() -> None:
    quotes = synthetic_ssvi_quotes(rho=-0.6, eta=0.8, gamma=0.3, noise=0.0)
    surface = VolSurface.fit(quotes)

    print(f"报价 {len(quotes)} 条，{surface.n_slices} 个到期日")
    print(f"{'T':>7}{'ATM':>9}{'k=-0.3':>10}{'k=+0.3':>10}")
    for t in surface.maturities:
        print(f"{t:>7.2f}{surface.atm_vol(t):>9.2%}"
              f"{float(np.asarray(surface.iv(-0.3, t))):>10.2%}"
              f"{float(np.asarray(surface.iv(0.3, t))):>10.2%}")

    report = surface_report(surface)
    print("\n无套利:", "是" if report["arbitrage_free"] else "否")
    print("日历 min Δw:", f"{report['calendar']['min_delta_w']:+.6f}")

    ks = np.linspace(-0.5, 0.5, 21)
    fit = SSVISurface.fit([(surface.atm_total_variance(t), ks,
                            np.asarray(surface.total_variance(ks, t)))
                           for t in surface.maturities])
    print(f"\nSSVI 拟合：rho={fit.rho:+.4f} eta={fit.eta:.4f} gamma={fit.gamma:.4f}")
    print("真参数：  rho=-0.6000 eta=0.8000 gamma=0.3000")


if __name__ == "__main__":
    main()