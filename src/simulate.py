"""2026 Dünya Kupası Monte Carlo turnuva simülatörü.

Format: 48 takım, 12 grup × 4. Her gruptan ilk 2 + en iyi 8 üçüncü → 32'li tur.
Sonra 16, çeyrek, yarı, final.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict


def _h2h_minitable(block: list, matches: list) -> dict:
    """Compute head-to-head mini-table for `block` teams.
    matches: list of (a, b, ga, gb). Only counts matches where both a,b in block.
    Returns {team: [Pts, GD, GF]}.
    """
    h2h = {t: [0, 0, 0] for t in block}
    for a, b, ga, gb in matches:
        if a in h2h and b in h2h:
            h2h[a][2] += ga
            h2h[b][2] += gb
            h2h[a][1] += ga - gb
            h2h[b][1] += gb - ga
            if ga > gb:
                h2h[a][0] += 3
            elif gb > ga:
                h2h[b][0] += 3
            else:
                h2h[a][0] += 1
                h2h[b][0] += 1
    return h2h


def _break_ties(block: list, overall: dict, matches: list) -> list:
    """Recursive H2H tiebreaker for tied `block`.
    overall[t] = [Pts, GD, GF, ...] (only GD, GF used as fallback).
    Order: H2H Pts → H2H GD → H2H GF → overall GD → overall GF.
    """
    if len(block) <= 1:
        return block
    h2h = _h2h_minitable(block, matches)
    block_sorted = sorted(block, key=lambda t: (-h2h[t][0], -h2h[t][1], -h2h[t][2]))
    result: list = []
    i = 0
    while i < len(block_sorted):
        j = i + 1
        key_i = tuple(h2h[block_sorted[i]])
        while j < len(block_sorted) and tuple(h2h[block_sorted[j]]) == key_i:
            j += 1
        sub = block_sorted[i:j]
        if len(sub) > 1:
            if len(sub) < len(block):
                # sub is strict subset → recurse, mini-table re-built over sub only
                sub = _break_ties(sub, overall, matches)
            else:
                # H2H gives identical (Pts, GD, GF) for whole block → overall fallback
                sub = sorted(sub, key=lambda t: (-overall[t][1], -overall[t][2]))
        result.extend(sub)
        i = j
    return result


def _rank_with_h2h(group: list, overall: dict, matches: list) -> list:
    """Rank `group` using FIFA tiebreaker order.
    Step 1: sort by Pts.  Step 2: for each Pts-block, apply H2H break.
    """
    initial = sorted(group, key=lambda t: -overall[t][0])
    result: list = []
    i = 0
    while i < len(initial):
        j = i + 1
        while j < len(initial) and overall[initial[j]][0] == overall[initial[i]][0]:
            j += 1
        block = initial[i:j]
        if len(block) > 1:
            block = _break_ties(block, overall, matches)
        result.extend(block)
        i = j
    return result


def _outcome_sample(rng: np.random.Generator, p_w: float, p_d: float,
                    p_l: float) -> int:
    """0=home win, 1=draw, 2=away win."""
    r = rng.random()
    if r < p_w:
        return 0
    if r < p_w + p_d:
        return 1
    return 2


def _ko_winner(rng: np.random.Generator, elo, a: str, b: str) -> str:
    """Eleme — beraberlik penaltıya gider, basit ~50/50 ama Elo'ya hafif eğim."""
    p_w, p_d, p_l = elo.predict_proba(a, b, neutral=True)
    out = _outcome_sample(rng, p_w, p_d, p_l)
    if out == 0:
        return a
    if out == 2:
        return b
    # beraberlik → Elo farkına göre penaltı
    rh, ra = elo.get(a), elo.get(b)
    p = 1.0 / (1.0 + 10 ** (-(rh - ra) / 800))  # daha yumuşak
    return a if rng.random() < p else b


def _make_groups(teams: list[str], rng: np.random.Generator,
                 pots: list[list[str]] | None = None) -> list[list[str]]:
    """12 grup × 4. Pot yoksa rastgele dağıt."""
    if pots is None:
        order = list(teams)
        rng.shuffle(order)
        return [order[i * 4:(i + 1) * 4] for i in range(12)]
    # pot tabanlı: her potta 12 takım, her grupta 1 pot temsilcisi
    groups = [[] for _ in range(12)]
    for pot in pots:
        idx = list(range(12))
        rng.shuffle(idx)
        for i, team in zip(idx, pot):
            groups[i].append(team)
    return groups


def _ko_winner_cb(rng: np.random.Generator, proba_fn, goals_fn, a: str, b: str
                   ) -> str:
    p_w, p_d, p_l = proba_fn(a, b, True)
    out = _outcome_sample(rng, p_w, p_d, p_l)
    if out == 0:
        return a
    if out == 2:
        return b
    # beraberlik → goals_fn ile penaltı simulasyonu basitçe ~50/50
    return a if rng.random() < (p_w + 0.5 * p_d) / (p_w + p_l + 1e-9) else b


def _play_group_cb(rng: np.random.Generator, proba_fn, goals_fn,
                   group: list[str]) -> pd.DataFrame:
    standings = {t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0,
                     "GF": 0, "GA": 0, "GD": 0, "Pts": 0} for t in group}
    matches: list = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            ga, gb = goals_fn(rng, a, b, True)
            matches.append((a, b, ga, gb))
            standings[a]["GF"] += ga; standings[a]["GA"] += gb
            standings[b]["GF"] += gb; standings[b]["GA"] += ga
            if ga > gb:
                standings[a]["W"] += 1; standings[a]["Pts"] += 3
                standings[b]["L"] += 1
            elif ga < gb:
                standings[b]["W"] += 1; standings[b]["Pts"] += 3
                standings[a]["L"] += 1
            else:
                standings[a]["D"] += 1; standings[a]["Pts"] += 1
                standings[b]["D"] += 1; standings[b]["Pts"] += 1
    for s in standings.values():
        s["P"] = s["W"] + s["D"] + s["L"]
        s["GD"] = s["GF"] - s["GA"]
    overall = {t: [standings[t]["Pts"], standings[t]["GD"], standings[t]["GF"]]
               for t in group}
    ranked = _rank_with_h2h(group, overall, matches)
    df = pd.DataFrame([standings[t] for t in ranked]).reset_index(drop=True)
    return df


def simulate_tournament_cb(proba_fn, goals_fn,
                            fixed_groups: list[list[str]],
                            rng: np.random.Generator) -> dict:
    """Generic simulator — proba_fn(h,a,neutral)->(pw,pd,pl),
    goals_fn(rng,h,a,neutral)->(gh,ga).
    fixed_groups: 12 lists ordered A..L.
    """
    first: list[str] = [None] * 12   # type: ignore[list-item]
    second: list[str] = [None] * 12  # type: ignore[list-item]
    third_candidates: list[tuple] = []
    for gi, g in enumerate(fixed_groups):
        st = _play_group_cb(rng, proba_fn, goals_fn, g)
        first[gi] = st.iloc[0]["team"]
        second[gi] = st.iloc[1]["team"]
        third_candidates.append((
            st.iloc[2]["Pts"], st.iloc[2]["GD"], st.iloc[2]["GF"],
            st.iloc[2]["team"], gi,
        ))
    third_candidates.sort(reverse=True)
    thirds_by_group = {gi: t for (_, _, _, t, gi) in third_candidates[:8]}
    knockout = _build_r32(first, second, thirds_by_group)
    rounds_results = {"R32": list(knockout)}
    for round_name in ["R16", "QF", "SF", "F"]:
        next_round = []
        for i in range(0, len(knockout), 2):
            w = _ko_winner_cb(rng, proba_fn, goals_fn,
                              knockout[i], knockout[i + 1])
            next_round.append(w)
        rounds_results[round_name] = list(next_round)
        knockout = next_round
        if len(knockout) == 1:
            break
    champ = (_ko_winner_cb(rng, proba_fn, goals_fn, knockout[0], knockout[1])
             if len(knockout) == 2 else knockout[0])
    return {"champion": champ, "rounds": rounds_results}


# Official 2026 R32 bracket — source: FIFA regulations + Wikipedia knockout stage
# Group index: A=0 B=1 C=2 D=3 E=4 F=5 G=6 H=7 I=8 J=9 K=10 L=11
#
# Slot type: "W"=winner, "RU"=runner-up, "3rd"=third-place
# For "3rd" slots, the set contains eligible group indices.
#
# Bracket order determines ALL subsequent rounds via consecutive pairing:
#   R32  positions (0,1),(2,3),...,(30,31) → 16 matches
#   R16  R32 winners (0-1 vs 2-3),(4-5 vs 6-7),...,(28-29 vs 30-31) → 8 matches
#   QF   R16 winners in same fashion → 4 matches, SF → 2, Final → 1
#
# R16 pairings verified against official schedule:
#   W(M74) vs W(M77) | W(M73) vs W(M75) | W(M76) vs W(M78) | W(M79) vs W(M80)
#   W(M83) vs W(M84) | W(M81) vs W(M82) | W(M86) vs W(M88) | W(M85) vs W(M87)

_SLOT_W   = 0   # group winner
_SLOT_RU  = 1   # runner-up
_SLOT_3RD = 2   # best 3rd-place; value = frozenset of eligible group indices

_BRACKET_DEF: list[tuple] = [
    # ── R16 pod 1: W(M74) vs W(M77) ──────────────────────────────────────────
    (_SLOT_W,   4,  _SLOT_3RD, frozenset({0,1,2,3,5})),     # M74: 1E vs 3rd(A/B/C/D/F)
    (_SLOT_W,   8,  _SLOT_3RD, frozenset({2,3,5,6,7})),     # M77: 1I vs 3rd(C/D/F/G/H)
    # ── R16 pod 2: W(M73) vs W(M75) ──────────────────────────────────────────
    (_SLOT_RU,  0,  _SLOT_RU,  1),                           # M73: 2A vs 2B
    (_SLOT_W,   5,  _SLOT_RU,  2),                           # M75: 1F vs 2C
    # ── R16 pod 3: W(M76) vs W(M78) ──────────────────────────────────────────
    (_SLOT_W,   2,  _SLOT_RU,  5),                           # M76: 1C vs 2F
    (_SLOT_RU,  4,  _SLOT_RU,  8),                           # M78: 2E vs 2I
    # ── R16 pod 4: W(M79) vs W(M80) ──────────────────────────────────────────
    (_SLOT_W,   0,  _SLOT_3RD, frozenset({2,4,5,7,8})),     # M79: 1A vs 3rd(C/E/F/H/I)
    (_SLOT_W,  11,  _SLOT_3RD, frozenset({4,7,8,9,10})),    # M80: 1L vs 3rd(E/H/I/J/K)
    # ── R16 pod 5: W(M83) vs W(M84) ──────────────────────────────────────────
    (_SLOT_RU, 10,  _SLOT_RU, 11),                           # M83: 2K vs 2L
    (_SLOT_W,   7,  _SLOT_RU,  9),                           # M84: 1H vs 2J
    # ── R16 pod 6: W(M81) vs W(M82) ──────────────────────────────────────────
    (_SLOT_W,   3,  _SLOT_3RD, frozenset({1,4,5,8,9})),     # M81: 1D vs 3rd(B/E/F/I/J)
    (_SLOT_W,   6,  _SLOT_3RD, frozenset({0,4,7,8,9})),     # M82: 1G vs 3rd(A/E/H/I/J)
    # ── R16 pod 7: W(M86) vs W(M88) ──────────────────────────────────────────
    (_SLOT_W,   9,  _SLOT_RU,  7),                           # M86: 1J vs 2H
    (_SLOT_RU,  3,  _SLOT_RU,  6),                           # M88: 2D vs 2G
    # ── R16 pod 8: W(M85) vs W(M87) ──────────────────────────────────────────
    (_SLOT_W,   1,  _SLOT_3RD, frozenset({4,5,6,8,9})),     # M85: 1B vs 3rd(E/F/G/I/J)
    (_SLOT_W,  10,  _SLOT_3RD, frozenset({3,4,8,9,11})),    # M87: 1K vs 3rd(D/E/I/J/L)
]


def _assign_thirds(
    thirds_by_group: dict,          # {group_idx: team_name} for the 8 qualifying thirds
    eligible_sets: list,            # list of frozensets, one per 3rd slot (8 slots)
) -> list:
    """Backtracking assignment: each 3rd-place team → exactly one eligible slot.
    Returns list of team names aligned with eligible_sets order."""
    items = list(thirds_by_group.items())  # [(gi, name), ...]
    result: list = [None] * len(eligible_sets)
    used: set = set()

    def bt(slot: int) -> bool:
        if slot == len(eligible_sets):
            return True
        for gi, name in items:
            if gi in eligible_sets[slot] and gi not in used:
                result[slot] = name
                used.add(gi)
                if bt(slot + 1):
                    return True
                used.remove(gi)
        return False

    bt(0)
    return result


def _build_r32(
    first: list,
    second: list,
    thirds_by_group: dict,
) -> list:
    """Build ordered 32-team bracket list.
    first[gi] = group winner, second[gi] = runner-up, thirds_by_group = {gi: team}.
    """
    eligible_sets = [
        slot[3]
        for slot in _BRACKET_DEF
        if slot[0] == _SLOT_W and slot[2] == _SLOT_3RD
    ]
    assigned = _assign_thirds(thirds_by_group, eligible_sets)
    third_iter = iter(assigned)

    bracket: list = []
    for slot in _BRACKET_DEF:
        ta_type, ta_val, tb_type, tb_val = slot
        # Team A
        if ta_type == _SLOT_W:
            bracket.append(first[ta_val])
        elif ta_type == _SLOT_RU:
            bracket.append(second[ta_val])
        else:
            bracket.append(next(third_iter))
        # Team B
        if tb_type == _SLOT_W:
            bracket.append(first[tb_val])
        elif tb_type == _SLOT_RU:
            bracket.append(second[tb_val])
        else:
            bracket.append(next(third_iter))
    return bracket


def build_cache(proba_fn, goals_fn, teams: list[str],
                max_goals: int = 8) -> tuple[dict, dict]:
    """Precompute proba + score matrix for all ordered team pairs (neutral).

    proba_cache[(a,b)] = (pw, pd, pl)
    score_cache[(a,b)] = flat normalized score matrix (flattened (max_goals+1)^2 array)
    """
    proba_cache: dict = {}
    score_cache: dict = {}
    mg1 = max_goals + 1
    for a in teams:
        for b in teams:
            if a == b:
                continue
            proba_cache[(a, b)] = proba_fn(a, b, True)
            # goals_fn not directly cacheable — use DC score_matrix if available
            # fallback: derive flat pmf from proba
    # score_cache built externally if dc available; otherwise use proba only
    return proba_cache, score_cache


def _ko_winner_cached(rng: np.random.Generator, a: str, b: str,
                       proba_cache: dict, score_cache: dict) -> str:
    p_w, p_d, p_l = proba_cache[(a, b)]
    out = _outcome_sample(rng, p_w, p_d, p_l)
    if out == 0:
        return a
    if out == 2:
        return b
    return a if rng.random() < (p_w + 0.5 * p_d) / (p_w + p_l + 1e-9) else b


def _sample_score_cached(rng: np.random.Generator, a: str, b: str,
                          score_cache: dict, proba_cache: dict) -> tuple[int, int]:
    key = (a, b)
    if key in score_cache:
        flat = score_cache[key]
        idx = rng.choice(len(flat), p=flat)
        mg1 = int(len(flat) ** 0.5)
        return int(idx // mg1), int(idx % mg1)
    # fallback: derive from proba
    p_w, p_d, p_l = proba_cache[key]
    out = _outcome_sample(rng, p_w, p_d, p_l)
    if out == 0:
        return 1, 0
    if out == 2:
        return 0, 1
    return 1, 1


def _play_group_cached(rng: np.random.Generator, group: list[str],
                        proba_cache: dict, score_cache: dict) -> tuple:
    """Plays all 6 group matches. Returns (ranked_teams, st_dict).
    Ranking uses FIFA tiebreaker: Pts → H2H Pts → H2H GD → H2H GF → overall GD → GF.
    """
    st = {t: [0, 0, 0, 0, 0] for t in group}  # [Pts, GD, GF, W, D]
    matches: list = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            ga, gb = _sample_score_cached(rng, a, b, score_cache, proba_cache)
            matches.append((a, b, ga, gb))
            st[a][2] += ga; st[b][2] += gb
            st[a][1] += ga - gb; st[b][1] += gb - ga
            if ga > gb:
                st[a][0] += 3; st[a][3] += 1
            elif ga < gb:
                st[b][0] += 3; st[b][3] += 1
            else:
                st[a][0] += 1; st[a][4] += 1
                st[b][0] += 1; st[b][4] += 1
    return _rank_with_h2h(group, st, matches), st


def simulate_tournament_cached(rng: np.random.Generator,
                                fixed_groups: list[list[str]],
                                proba_cache: dict, score_cache: dict) -> dict:
    """fixed_groups: 12 lists ordered A..L, each 4 teams (from load_groups_2026)."""
    first: list[str] = [None] * 12   # type: ignore[list-item]
    second: list[str] = [None] * 12  # type: ignore[list-item]
    third_candidates: list[tuple] = []
    for gi, g in enumerate(fixed_groups):
        ranked, st = _play_group_cached(rng, g, proba_cache, score_cache)
        first[gi] = ranked[0]
        second[gi] = ranked[1]
        t3 = ranked[2]
        third_candidates.append((st[t3][0], st[t3][1], st[t3][2], t3, gi))
    third_candidates.sort(reverse=True)
    thirds_by_group = {gi: t for (_, _, _, t, gi) in third_candidates[:8]}
    knockout = _build_r32(first, second, thirds_by_group)
    rounds_results = {"R32": list(knockout)}
    for round_name in ["R16", "QF", "SF", "F"]:
        next_round = []
        for i in range(0, len(knockout), 2):
            w = _ko_winner_cached(rng, knockout[i], knockout[i + 1],
                                  proba_cache, score_cache)
            next_round.append(w)
        rounds_results[round_name] = list(next_round)
        knockout = next_round
        if len(knockout) == 1:
            break
    champ = (_ko_winner_cached(rng, knockout[0], knockout[1], proba_cache, score_cache)
             if len(knockout) == 2 else knockout[0])
    return {"champion": champ, "rounds": rounds_results}


def run_monte_carlo_cached(
    proba_cache: dict, score_cache: dict,
    fixed_groups: list[list[str]],
    n: int = 10000, seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fast MC using precomputed proba/score caches. Avoids model calls in loop.
    fixed_groups: 12 lists ordered A..L (from load_groups_2026).
    """
    teams = [t for g in fixed_groups for t in g]

    def _batch(n_local: int, seed_local: int) -> dict:
        rng = np.random.default_rng(seed_local)
        counts: dict = {k: defaultdict(int)
                        for k in ["champion", "R32", "R16", "QF", "SF", "F"]}
        for _ in range(n_local):
            res = simulate_tournament_cached(rng, fixed_groups, proba_cache, score_cache)
            counts["champion"][res["champion"]] += 1
            for rnd, ts in res["rounds"].items():
                for t in ts:
                    counts[rnd][t] += 1
        return counts

    if n_jobs > 1:
        from joblib import Parallel, delayed
        chunk = max(1, n // n_jobs)
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_batch)(chunk, seed + i) for i in range(n_jobs)
        )
        merged: dict = {k: defaultdict(int)
                        for k in ["champion", "R32", "R16", "QF", "SF", "F"]}
        total = 0
        for r in results:
            total += sum(r["champion"].values())
            for key in merged:
                for t, v in r[key].items():
                    merged[key][t] += v
        counts, n = merged, total
    else:
        counts = _batch(n, seed)

    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "P_R32":      counts["R32"][t] / n,
            "P_R16":      counts["R16"][t] / n,
            "P_QF":       counts["QF"][t] / n,
            "P_SF":        counts["SF"][t] / n,
            "P_Final":    counts["F"][t] / n,
            "P_Champion": counts["champion"][t] / n,
        })
    return (pd.DataFrame(rows)
            .sort_values("P_Champion", ascending=False)
            .reset_index(drop=True))


def _run_batch(args):
    """Picklable worker for parallel MC."""
    proba_fn, goals_fn, fixed_groups, n, seed = args
    rng = np.random.default_rng(seed)
    counts: dict = {"champion": defaultdict(int)}
    for rnd in ["R32", "R16", "QF", "SF", "F"]:
        counts[rnd] = defaultdict(int)
    for _ in range(n):
        res = simulate_tournament_cb(proba_fn, goals_fn, fixed_groups, rng)
        counts["champion"][res["champion"]] += 1
        for rnd, ts in res["rounds"].items():
            for t in ts:
                counts[rnd][t] += 1
    return counts


def run_monte_carlo_cb(
    proba_fn, goals_fn,
    fixed_groups: list[list[str]],
    n: int = 10000, seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    teams = [t for g in fixed_groups for t in g]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        chunk = max(1, n // n_jobs)
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_batch)((proba_fn, goals_fn, fixed_groups, chunk, seed + i))
            for i in range(n_jobs)
        )
        merged: dict = {"champion": defaultdict(int)}
        for rnd in ["R32", "R16", "QF", "SF", "F"]:
            merged[rnd] = defaultdict(int)
        total = 0
        for r in results:
            total += sum(r["champion"].values())
            for key in merged:
                for t, v in r[key].items():
                    merged[key][t] += v
        counts, n = merged, total
    else:
        rng = np.random.default_rng(seed)
        counts: dict = defaultdict(lambda: defaultdict(int))  # type: ignore[assignment]
        for _ in range(n):
            res = simulate_tournament_cb(proba_fn, goals_fn, fixed_groups, rng)
            counts["champion"][res["champion"]] += 1
            for rnd, ts in res["rounds"].items():
                for t in ts:
                    counts[rnd][t] += 1
    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "P_R32":      counts["R32"][t] / n,
            "P_R16":      counts["R16"][t] / n,
            "P_QF":       counts["QF"][t] / n,
            "P_SF":        counts["SF"][t] / n,
            "P_Final":    counts["F"][t] / n,
            "P_Champion": counts["champion"][t] / n,
        })
    return (pd.DataFrame(rows)
            .sort_values("P_Champion", ascending=False)
            .reset_index(drop=True))


def _play_group(rng: np.random.Generator, elo, group: list[str]) -> pd.DataFrame:
    standings = {t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0,
                     "GF": 0, "GA": 0, "GD": 0, "Pts": 0} for t in group}
    matches: list = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            p_w, p_d, p_l = elo.predict_proba(a, b, neutral=True)
            out = _outcome_sample(rng, p_w, p_d, p_l)
            rh, ra = elo.get(a), elo.get(b)
            mu_a = max(0.4, 1.4 + (rh - ra) / 400)
            mu_b = max(0.4, 1.4 - (rh - ra) / 400)
            ga, gb = rng.poisson(mu_a), rng.poisson(mu_b)
            if out == 0 and ga <= gb:
                ga = gb + 1
            elif out == 2 and gb <= ga:
                gb = ga + 1
            elif out == 1 and ga != gb:
                gb = ga
            matches.append((a, b, int(ga), int(gb)))
            standings[a]["GF"] += ga; standings[a]["GA"] += gb
            standings[b]["GF"] += gb; standings[b]["GA"] += ga
            if ga > gb:
                standings[a]["W"] += 1; standings[a]["Pts"] += 3
                standings[b]["L"] += 1
            elif ga < gb:
                standings[b]["W"] += 1; standings[b]["Pts"] += 3
                standings[a]["L"] += 1
            else:
                standings[a]["D"] += 1; standings[a]["Pts"] += 1
                standings[b]["D"] += 1; standings[b]["Pts"] += 1
    for s in standings.values():
        s["P"] = s["W"] + s["D"] + s["L"]
        s["GD"] = s["GF"] - s["GA"]
    overall = {t: [standings[t]["Pts"], standings[t]["GD"], standings[t]["GF"]]
               for t in group}
    ranked = _rank_with_h2h(group, overall, matches)
    df = pd.DataFrame([standings[t] for t in ranked]).reset_index(drop=True)
    return df


def simulate_tournament(elo, fixed_groups: list[list[str]],
                         rng: np.random.Generator) -> dict:
    """Elo-only (S model) simulator. fixed_groups: 12 lists ordered A..L."""
    first: list[str] = [None] * 12   # type: ignore[list-item]
    second: list[str] = [None] * 12  # type: ignore[list-item]
    third_candidates: list[tuple] = []
    for gi, g in enumerate(fixed_groups):
        st = _play_group(rng, elo, g)
        first[gi] = st.iloc[0]["team"]
        second[gi] = st.iloc[1]["team"]
        third_candidates.append((
            st.iloc[2]["Pts"], st.iloc[2]["GD"], st.iloc[2]["GF"],
            st.iloc[2]["team"], gi,
        ))
    third_candidates.sort(reverse=True)
    thirds_by_group = {gi: t for (_, _, _, t, gi) in third_candidates[:8]}
    knockout = _build_r32(first, second, thirds_by_group)
    rounds_results = {"R32": list(knockout)}
    for round_name in ["R16", "QF", "SF", "F"]:
        next_round = []
        for i in range(0, len(knockout), 2):
            w = _ko_winner(rng, elo, knockout[i], knockout[i + 1])
            next_round.append(w)
        rounds_results[round_name] = list(next_round)
        knockout = next_round
        if len(knockout) == 1:
            break
    champ = (_ko_winner(rng, elo, knockout[0], knockout[1])
             if len(knockout) == 2 else knockout[0])
    return {"champion": champ, "rounds": rounds_results}


def run_monte_carlo(
    elo,
    fixed_groups: list[list[str]],
    n: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    teams = [t for g in fixed_groups for t in g]
    rng = np.random.default_rng(seed)
    counts = defaultdict(lambda: defaultdict(int))
    for _ in range(n):
        res = simulate_tournament(elo, fixed_groups, rng)
        counts["champion"][res["champion"]] += 1
        for rnd, ts in res["rounds"].items():
            for t in ts:
                counts[rnd][t] += 1
    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "P_R32":      counts["R32"][t] / n,
            "P_R16":      counts["R16"][t] / n,
            "P_QF":       counts["QF"][t] / n,
            "P_SF":        counts["SF"][t] / n,
            "P_Final":    counts["F"][t] / n,
            "P_Champion": counts["champion"][t] / n,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("P_Champion", ascending=False)
        .reset_index(drop=True)
    )
