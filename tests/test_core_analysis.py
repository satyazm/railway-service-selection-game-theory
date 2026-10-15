import math

from src.characteristic_function import coalition_key, compute_characteristic_function
from src.core_analysis import core_violations, least_core, lp_dual_allocation
from src.milp import solve_service_selection_milp
from src.network import load_all


def test_least_core_matches_classic_majority_game():
    """3-player majority game: v(S) = 1 if |S| >= 2 else 0. The core is a
    textbook example of an empty core; the least-core epsilon has the known
    closed form 1/3 (symmetric allocation x_i = 1/3 each)."""
    players = ["1", "2", "3"]
    v = {
        (): 0.0, ("1",): 0.0, ("2",): 0.0, ("3",): 0.0,
        ("1", "2"): 1.0, ("1", "3"): 1.0, ("2", "3"): 1.0,
        ("1", "2", "3"): 1.0,
    }
    result = least_core(players, v)
    assert not result["in_core"]
    assert math.isclose(result["epsilon"], 1 / 3, abs_tol=1e-5)
    for i in players:
        assert math.isclose(result["allocation"][i], 1 / 3, abs_tol=1e-5)


def test_least_core_detects_nonempty_core():
    players = ["a", "b"]
    v = {(): 0.0, ("a",): 0.0, ("b",): 0.0, ("a", "b"): 1.0}
    result = least_core(players, v)
    assert result["in_core"]
    assert result["epsilon"] <= 1e-6
    assert math.isclose(result["allocation"]["a"] + result["allocation"]["b"], 1.0, abs_tol=1e-6)
    assert core_violations(players, v, result["allocation"]) == []


def test_core_violations_detects_unstable_allocation():
    players = ["1", "2", "3"]
    v = {
        (): 0.0, ("1",): 0.0, ("2",): 0.0, ("3",): 0.0,
        ("1", "2"): 1.0, ("1", "3"): 1.0, ("2", "3"): 1.0,
        ("1", "2", "3"): 1.0,
    }
    # equal split satisfies every pairwise constraint exactly (1/3+1/3=2/3 < 1)
    violations = core_violations(players, v, {"1": 1 / 3, "2": 1 / 3, "3": 1 / 3})
    assert len(violations) == 3
    for (coalition, shortfall) in violations:
        assert len(coalition) == 2
        assert math.isclose(shortfall, 1 / 3, abs_tol=1e-5)


def test_least_core_allocation_is_efficient():
    players = ["1", "2", "3"]
    v = {
        (): 0.0, ("1",): 0.0, ("2",): 0.0, ("3",): 0.0,
        ("1", "2"): 1.0, ("1", "3"): 1.0, ("2", "3"): 1.0,
        ("1", "2", "3"): 1.0,
    }
    result = least_core(players, v)
    assert math.isclose(sum(result["allocation"].values()), v[("1", "2", "3")], abs_tol=1e-5)


def test_real_game_core_is_empty_due_to_strong_substitution():
    """On the real Chennai-Bangalore corridor, Shatabdi alone already
    captures the large majority of the grand coalition's value (services
    heavily substitute for each other on the same corridor), so no
    allocation can be simultaneously fair and coalition-proof: the exact
    core is empty."""
    segments, services, od_df = load_all()
    players = list(services.keys())
    v, _ = compute_characteristic_function(services, segments, od_df)
    result = least_core(players, v)
    assert not result["in_core"]
    assert result["epsilon"] > 0


def test_lp_dual_allocation_is_efficient_and_duality_holds():
    segments, services, od_df = load_all()
    grand = solve_service_selection_milp(services, segments, od_df, coalition=None)
    dual = lp_dual_allocation(services, segments, od_df, grand)
    assert math.isclose(dual["total_allocated"], grand.objective, rel_tol=1e-4)
    assert math.isclose(dual["duality_gap"], 0.0, abs_tol=1.0)
    assert all(val >= -1e-6 for val in dual["allocation"].values())
