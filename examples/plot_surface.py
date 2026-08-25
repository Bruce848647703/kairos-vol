"""生成 README 用的曲面图（需要 matplotlib：pip install -e ".[plot]"）。

运行：python examples/plot_surface.py
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kairos_vol.data import synthetic_ssvi_quotes  # noqa: E402
from kairos_vol.surface import VolSurface  # noqa: E402


def main(out: str = "docs/vol-surface.png") -> None:
    surface = VolSurface.fit(synthetic_ssvi_quotes())
    ks = np.linspace(-0.5, 0.5, 61)
    ts = np.linspace(surface.maturities[0], surface.maturities[-1], 40)
    grid = np.array([[float(surface.iv(k, t)) * 100.0 for k in ks] for t in ts])

    fig = plt.figure(figsize=(12, 4.6))

    ax1 = fig.add_subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(surface.maturities)))
    for color, t in zip(colors, surface.maturities):
        ivs = [float(surface.iv(k, t)) * 100.0 for k in ks]
        ax1.plot(ks, ivs, color=color, lw=1.8, label=f"T={t:g}y")
    ax1.set_xlabel("log-moneyness  k = ln(K/F)")
    ax1.set_ylabel("implied vol (%)")
    ax1.set_title("SVI smiles by maturity")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    kk, tt = np.meshgrid(ks, ts)
    ax2.plot_surface(kk, tt, grid, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_xlabel("k")
    ax2.set_ylabel("maturity (y)")
    ax2.set_zlabel("implied vol (%)")
    ax2.set_title("Volatility surface")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()