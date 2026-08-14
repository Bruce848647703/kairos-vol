"""示例 2：方差互换与 vol carry —— 曲面告诉你的「贵的波动率」。

运行：python examples/02_variance_carry.py
"""
from __future__ import annotations

import numpy as np

from kairos_vol.data import synthetic_ssvi_quotes
from kairos_vol.localvol import dupire_local_vol
from kairos_vol.surface import VolSurface
from kairos_vol.variance import fair_variance_strike, term_structure


def main() -> None:
    surface = VolSurface.fit(synthetic_ssvi_quotes(rho=-0.6, eta=0.8, gamma=0.3))

    print(f"{'T':>7}{'ATM vol':>10}{'公允 vol':>10}{'carry(var)':>13}{'σ_loc(ATM)':>12}")
    print("-" * 54)
    for row in term_structure(surface):
        t = row["maturity"]
        local = dupire_local_vol(surface, 0.0, t)
        print(f"{t:>7.2f}{row['atm_vol']:>10.2%}{row['fair_vol']:>10.2%}"
              f"{row['carry_variance']:>+13.5f}{local:>12.2%}")

    t = 1.0
    fair_var = fair_variance_strike(surface, t)
    atm_var = surface.atm_vol(t) ** 2
    print(f"\nT=1：ATM 方差 {atm_var:.5f} vs 公允方差 {fair_var:.5f} "
          f"=> 买入方差互换的 carry 为 {fair_var - atm_var:+.5f}")
    print("（负偏斜越强、微笑越凸，公允方差越高于平值方差 —— 这就是做空平值波动率的成本）")


if __name__ == "__main__":
    main()