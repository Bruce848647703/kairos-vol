"""数据与命令行测试。"""
from __future__ import annotations

import numpy as np
import pytest

from kairos_vol.cli import main
from kairos_vol.data import (
    load_quotes_csv,
    save_quotes_csv,
    surface_from_quotes,
    synthetic_ssvi_quotes,
)


def test_synthetic_quotes_shape_and_values():
    quotes = synthetic_ssvi_quotes(maturities=(0.25, 1.0), n_k=11)
    assert len(quotes) == 22
    assert all(q.maturity in (0.25, 1.0) for q in quotes)
    assert all(0 < q.iv < 1 for q in quotes)
    assert all(q.forward == pytest.approx(100.0) for q in quotes)


def test_csv_roundtrip(tmp_path):
    quotes = synthetic_ssvi_quotes(maturities=(0.5,), n_k=9)
    path = tmp_path / "quotes.csv"
    save_quotes_csv(quotes, str(path))
    loaded = load_quotes_csv(str(path))
    assert len(loaded) == len(quotes)
    for a, b in zip(quotes, loaded):
        assert a.maturity == b.maturity
        assert a.strike == pytest.approx(b.strike)
        assert a.iv == pytest.approx(b.iv)
        assert a.forward == pytest.approx(b.forward)


def test_csv_column_aliases(tmp_path):
    path = tmp_path / "alias.csv"
    path.write_text("t,K,vol,F\n0.5,100,0.2,100\n0.5,110,0.19,100\n", encoding="utf-8")
    quotes = load_quotes_csv(str(path))
    assert len(quotes) == 2
    assert quotes[0].iv == pytest.approx(0.2)


def test_surface_from_quotes_helper():
    quotes = synthetic_ssvi_quotes(maturities=(0.25, 1.0), n_k=15)
    surface = surface_from_quotes(quotes)
    assert surface.n_slices == 2


def test_cli_price_and_iv(capsys):
    assert main(["price", "--spot", "100", "--strike", "100", "--vol", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "价格" in out and "Delta" in out

    assert main(["iv", "--price", "10", "--forward", "100", "--strike", "100", "--t", "1"]) == 0
    out = capsys.readouterr().out
    assert "隐含波动率" in out


def test_cli_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "无套利体检" in out
    assert "方差互换" in out
    assert "Dupire" in out


def test_cli_calibrate(tmp_path, capsys):
    from kairos_vol.data import save_quotes_csv

    path = tmp_path / "q.csv"
    save_quotes_csv(synthetic_ssvi_quotes(maturities=(0.25, 1.0), n_k=21), str(path))
    assert main(["calibrate", "--csv", str(path)]) == 0
    assert "无套利" in capsys.readouterr().out

    assert main(["surface", "--csv", str(path), "--t", "1.0"]) == 0
    assert "T\\k" in capsys.readouterr().out