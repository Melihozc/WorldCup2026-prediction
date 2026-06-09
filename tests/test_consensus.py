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


def _toy_groups():
    # 12 groups x 4 = 48 teams named T00..T47
    teams = [f"T{i:02d}" for i in range(48)]
    return [teams[i * 4:(i + 1) * 4] for i in range(12)]


def test_infer_market_abilities_recovers_rank_and_reduces_kl():
    groups = _toy_groups()
    teams = [t for g in groups for t in g]
    # synthetic market: geometric decay so ranks are unambiguous
    raw = np.array([0.97 ** i for i in range(len(teams))])
    market = pd.DataFrame({"team": teams, "market_prob": raw / raw.sum()})
    strengths, info = consensus.infer_market_abilities(
        market, groups, n_infer=3000, n_iter=8, lr=0.6, seed=1, n_jobs=1)
    assert set(strengths) == set(teams)
    # KL to market should drop over iterations
    assert info["kl_history"][-1] < info["kl_history"][0]
    # stronger market teams get higher inferred ability (rank agreement)
    mk = market.set_index("team")["market_prob"]
    s = pd.Series(strengths)
    corr = np.corrcoef(mk.rank(), s.reindex(mk.index).rank())[0, 1]
    assert corr > 0.9


def test_zscore_and_blend():
    z = consensus._zscore_dict({"A": 10.0, "B": 0.0, "C": -10.0})
    assert z["A"] > z["B"] > z["C"]
    assert abs(np.mean(list(z.values()))) < 1e-9
    blended = consensus.blend_strengths(
        {"m": {"A": 1.0, "B": -1.0}, "h": {"A": 2.0, "B": -2.0}},
        {"m": 0.5, "h": 0.5})
    assert blended["A"] == pytest.approx(1.5)
    assert blended["B"] == pytest.approx(-1.5)


def test_blend_handles_missing_component_team():
    blended = consensus.blend_strengths(
        {"m": {"A": 1.0, "B": 1.0}, "h": {"A": 3.0}},  # B missing in h
        {"m": 1.0, "h": 1.0})
    # B: (1*1 + 1*0)/2 = 0.5 ; A: (1+3)/2 = 2.0
    assert blended["A"] == pytest.approx(2.0)
    assert blended["B"] == pytest.approx(0.5)
