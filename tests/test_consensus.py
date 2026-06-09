import numpy as np
import pandas as pd
import pytest

from src import consensus


def _write_odds(tmp_path, name, rows):
    p = tmp_path / name
    pd.DataFrame(rows, columns=["team", "decimal_odds", "book", "date", "source"]).to_csv(p, index=False)
    return p


def test_load_consensus_two_books_averages_and_normalizes(tmp_path):
    a = _write_odds(tmp_path, "wc2026_outright_2026-06-01.csv", [
        ["Spain", 5.0, "A", "2026-06-01", "x"],
        ["France", 6.0, "A", "2026-06-01", "x"],
        ["Brazil", 8.0, "A", "2026-06-01", "x"],
    ])
    b = _write_odds(tmp_path, "wc2026_outright_2026-06-02.csv", [
        ["Spain", 7.0, "B", "2026-06-02", "x"],
        ["France", 5.0, "B", "2026-06-02", "x"],
        ["Brazil", 9.0, "B", "2026-06-02", "x"],
    ])
    out = consensus.load_consensus_odds([a, b])
    assert list(out.columns) == ["team", "market_prob"]
    assert set(out["team"]) == {"Spain", "France", "Brazil"}
    assert out["market_prob"].sum() == pytest.approx(1.0, abs=1e-9)
    probs = dict(zip(out["team"], out["market_prob"]))
    assert probs["Spain"] > probs["Brazil"]
    assert probs["France"] > probs["Brazil"]
