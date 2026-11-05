"""命令行入口：``kairos-vol`` / ``python -m kairos_vol``。"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np

from . import __version__
from .core import black76, black_scholes
from .implied import implied_vol
from .ssvi import SSVISurface
from .surface import VolSurface


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kairos-vol",
        description="波动率曲面研究：SVI/SSVI 校准、无套利检验、Dupire 局部波动率、方差互换。")
    parser.add_argument("--version", action="version", version=f"kairos-vol {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="合成曲面 -> 校准 -> 体检 -> 方差互换（默认）")

    price = sub.add_parser("price", help="Black-Scholes / Black-76 定价与希腊字母")
    price.add_argument("--spot", type=float, default=100.0)
    price.add_argument("--forward", type=float, default=None)
    price.add_argument("--strike", type=float, default=100.0)
    price.add_argument("--t", type=float, default=1.0)
    price.add_argument("--rate", type=float, default=0.02)
    price.add_argument("--dividend", type=float, default=0.0)
    price.add_argument("--vol", type=float, default=0.2)
    price.add_argument("--kind", default="call", choices=("call", "put"))

    iv = sub.add_parser("iv", help="由价格反解隐含波动率")
    iv.add_argument("--price", type=float, required=True)
    iv.add_argument("--forward", type=float, required=True)
    iv.add_argument("--strike", type=float, required=True)
    iv.add_argument("--t", type=float, required=True)
    iv.add_argument("--kind", default="call", choices=("call", "put"))

    cal = sub.add_parser("calibrate", help="从 CSV 校准曲面并做无套利体检")
    cal.add_argument("--csv", required=True)
    cal.add_argument("--k", type=float, default=0.0, help="查询该对数货币度的 IV")

    surf = sub.add_parser("surface", help="打印曲面网格")
    surf.add_argument("--csv", required=True)
    surf.add_argument("--k", type=float, default=0.0)
    surf.add_argument("--t", type=float, default=0.0, help="0 表示打印所有到期日")

    return parser


def _print_surface(surface: VolSurface) -> None:
    print(f"{'T':>8}{'ATM':>10}{'k=-.25':>10}{'k=+.25':>10}{'SVI(a,b,rho,m,sigma)':>44}")
    print("-" * 84)
    for t, sl in zip(surface.maturities, surface.slices):
        print(f"{t:>8.3f}{surface.atm_vol(t):>10.2%}"
              f"{float(np.asarray(sl.iv(-0.25, t))):>10.2%}"
              f"{float(np.asarray(sl.iv(0.25, t))):>10.2%}"
              f"   {sl.a:+.5f} {sl.b:.4f} {sl.rho:+.3f} {sl.m:+.3f} {sl.sigma:.4f}")


def _demo_surface() -> VolSurface:
    from .data import synthetic_ssvi_quotes

    quotes = synthetic_ssvi_quotes()
    return VolSurface.fit(quotes)


def _cmd_demo(_: argparse.Namespace) -> int:
    from .arbitrage import surface_report
    from .localvol import dupire_local_vol
    from .variance import term_structure
    from .data import synthetic_ssvi_quotes

    surface = _demo_surface()
    print("=" * 84)
    print("1) 合成 SSVI 报价 -> 逐期限 SVI 校准")
    print("=" * 84)
    _print_surface(surface)

    print()
    print("=" * 84)
    print("2) 无套利体检（蝶式 g(k)≥0 / 日历 w 随 T 不减）")
    print("=" * 84)
    report = surface_report(surface)
    print(f"  整体无套利: {'是' if report['arbitrage_free'] else '否'}")
    for key, row in report["butterfly"].items():
        print(f"  {key:<10} g_min={row['g_min']:+.6f}  w_min={row['w_min']:.6f}  "
              f"{'OK' if row['butterfly_free'] else 'VIOLATION'}")
    cal = report["calendar"]
    print(f"  日历: {'OK' if cal['calendar_free'] else 'VIOLATION'}  "
          f"min Δw={cal['min_delta_w']:+.6f}")

    print()
    print("=" * 84)
    print("3) SSVI 全局拟合（3 参数描述整条曲面）")
    print("=" * 84)
    ts = np.linspace(-0.5, 0.5, 21)
    slices = [(surface.atm_total_variance(t), ts,
               np.asarray(surface.total_variance(ts, t))) for t in surface.maturities]
    ssvi = SSVISurface.fit(slices)
    err = 0.0
    for theta, ks, ws in slices:
        err = max(err, float(np.max(np.abs(ssvi.w(ks, theta) - ws))))
    print(f"  rho={ssvi.rho:+.4f}  eta={ssvi.eta:.4f}  gamma={ssvi.gamma:.4f}")
    print(f"  最大总方差误差 = {err:.3e}   充分无套利条件: {'满足' if ssvi.no_arb_ok() else '不满足'}")

    print()
    print("=" * 84)
    print("4) 方差互换 / carry（对数合约复制）")
    print("=" * 84)
    print(f"{'T':>8}{'ATM vol':>10}{'公允 vol':>10}{'carry(var)':>14}")
    for row in term_structure(surface):
        print(f"{row['maturity']:>8.3f}{row['atm_vol']:>10.2%}{row['fair_vol']:>10.2%}"
              f"{row['carry_variance']:>+14.5f}")

    print()
    print("=" * 84)
    print("5) Dupire 局部波动率（ATM）")
    print("=" * 84)
    for t in surface.maturities:
        print(f"  T={t:>5.3f}  sigma_loc(k=0) = {dupire_local_vol(surface, 0.0, t):.2%}")
    print()
    print(f"（合成真参数：rho=-0.60 eta=0.80 gamma=0.30；样本 {len(synthetic_ssvi_quotes())} 条报价）")
    return 0


def _cmd_price(args: argparse.Namespace) -> int:
    if args.forward is not None:
        res = black76(args.forward, args.strike, args.t, args.vol, 1.0, args.kind)
        label = f"Black-76 F={args.forward:.4f}"
    else:
        res = black_scholes(args.spot, args.strike, args.t, args.rate, args.vol,
                            args.dividend, args.kind)
        label = f"Black-Scholes S={args.spot:.4f}"
    print(f"{label}  K={args.strike:.4f}  T={args.t:.4f}  σ={args.vol:.2%}  {args.kind}")
    print("-" * 52)
    print(f"  价格   {res['price']:.6f}")
    print(f"  Delta  {res['delta']:.6f}")
    print(f"  Gamma  {res['gamma']:.6f}")
    print(f"  Vega   {res['vega']:.6f}")
    print(f"  Theta  {res['theta']:.6f}")
    return 0


def _cmd_iv(args: argparse.Namespace) -> int:
    vol = implied_vol(args.price, args.forward, args.strike, args.t, 1.0, args.kind)
    print(f"隐含波动率 = {vol:.8f}  ({vol:.4%})")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .arbitrage import surface_report
    from .data import load_quotes_csv
    from .variance import term_structure

    quotes = load_quotes_csv(args.csv)
    surface = VolSurface.fit(quotes)
    print(f"读取 {len(quotes)} 条报价，{surface.n_slices} 个到期日")
    _print_surface(surface)
    report = surface_report(surface)
    print()
    print(f"无套利: {'是' if report['arbitrage_free'] else '否'}  "
          f"日历 min Δw={report['calendar']['min_delta_w']:+.6f}")
    print(f"k={args.k:+.3f} 处各期限 IV：")
    for t in surface.maturities:
        print(f"  T={t:>6.3f}  IV={float(np.asarray(surface.iv(args.k, t))):.4%}")
    print()
    print(f"{'T':>8}{'ATM':>10}{'公允 vol':>10}{'carry':>12}")
    for row in term_structure(surface):
        print(f"{row['maturity']:>8.3f}{row['atm_vol']:>10.2%}{row['fair_vol']:>10.2%}"
              f"{row['carry_variance']:>+12.5f}")
    return 0


def _cmd_surface(args: argparse.Namespace) -> int:
    from .data import load_quotes_csv

    surface = VolSurface.fit(load_quotes_csv(args.csv))
    ts = surface.maturities if args.t <= 0 else [args.t]
    ks = np.linspace(-0.5, 0.5, 11)
    header = "".join(f"{k:>8.2f}" for k in ks)
    label = "T\\k"
    print(f"{label:>8}{header}")
    for t in ts:
        row = "".join(f"{float(np.asarray(surface.iv(k, t))):>8.2%}" for k in ks)
        print(f"{t:>8.3f}{row}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not argv_list:
        argv_list = ["demo"]
    elif argv_list[0] not in ("-h", "--help", "--version") and argv_list[0].startswith("-"):
        # 无子命令时默认跑 demo；但 -h/--help/--version 必须留给顶层解析器
        argv_list = ["demo"] + argv_list
    args = parser.parse_args(argv_list)
    command = args.command or "demo"
    return {
        "demo": _cmd_demo,
        "price": _cmd_price,
        "iv": _cmd_iv,
        "calibrate": _cmd_calibrate,
        "surface": _cmd_surface,
    }[command](args)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())