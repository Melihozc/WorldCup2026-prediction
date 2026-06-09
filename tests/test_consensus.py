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


def test_strength_model_proba_and_lambdas():
    m = consensus.StrengthModel({"A": 1.0, "B": -1.0}, scale=1.0, base=0.262)
    lh, la = m.lambdas("A", "B", neutral=True)
    assert lh > la  # stronger team scores more
    pw, pd_, pl = m.proba("A", "B", neutral=True)
    assert pw > pl  # stronger team more likely to win
    assert pw + pd_ + pl == pytest.approx(1.0, abs=1e-9)
    # symmetry: equal strength → pw ~= pl on neutral
    me = consensus.StrengthModel({"A": 0.0, "B": 0.0}, scale=1.0)
    pw2, pd2, pl2 = me.proba("A", "B", neutral=True)
    assert pw2 == pytest.approx(pl2, abs=1e-9)


def test_strength_goals_fn_callable_with_simulate_signature():
    m = consensus.StrengthModel({"A": 0.5, "B": -0.5})
    gf = m.goals_fn()
    rng = np.random.default_rng(0)
    gh, ga = gf(rng, "A", "B", True)
    assert isinstance(gh, int) and isinstance(ga, int)
    pf = m.proba_fn()
    out = pf("A", "B", True)
    assert len(out) == 3
