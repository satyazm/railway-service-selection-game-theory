import math

from src.characteristic_function import coalition_key
from src.shapley import pairwise_complementarity, shapley_values, verify_efficiency


def _toy_game():
    """Small hand-checkable 3-player game.

    v({}) = 0, v({a}) = 1, v({b}) = 1, v({c}) = 1
    v({a,b}) = 4, v({a,c}) = 2, v({b,c}) = 2
    v({a,b,c}) = 9

    Shapley values computed by hand below are used as a ground truth.
    """
    v = {
        (): 0.0,
        ("a",): 1.0,
        ("b",): 1.0,
        ("c",): 1.0,
        ("a", "b"): 4.0,
        ("a", "c"): 2.0,
        ("b", "c"): 2.0,
        ("a", "b", "c"): 9.0,
    }
    return ["a", "b", "c"], v


def test_shapley_efficiency_axiom():
    players, v = _toy_game()
    phi = shapley_values(players, v)
    assert verify_efficiency(phi, v[coalition_key(players)])


def test_shapley_symmetry_axiom():
    # b and c are symmetric: swapping them leaves all coalition values with
    # {b} <-> {c} relabelled identical, so they must get equal Shapley value.
    players = ["a", "b", "c"]
    v = {
        (): 0.0,
        ("a",): 1.0,
        ("b",): 2.0,
        ("c",): 2.0,
        ("a", "b"): 5.0,
        ("a", "c"): 5.0,
        ("b", "c"): 6.0,
        ("a", "b", "c"): 10.0,
    }
    phi = shapley_values(players, v)
    assert math.isclose(phi["b"], phi["c"], abs_tol=1e-9)


def test_shapley_null_player_axiom():
    # d never contributes anything to any coalition -> phi[d] == 0
    players = ["a", "b", "d"]
    v = {
        (): 0.0,
        ("a",): 3.0,
        ("b",): 2.0,
        ("d",): 0.0,
        ("a", "b"): 7.0,
        ("a", "d"): 3.0,
        ("b", "d"): 2.0,
        ("a", "b", "d"): 7.0,
    }
    phi = shapley_values(players, v)
    assert math.isclose(phi["d"], 0.0, abs_tol=1e-9)


def test_shapley_matches_hand_computed_values():
    players, v = _toy_game()
    phi = shapley_values(players, v)
    # Hand-derived (by averaging marginal contributions over all 3! orders):
    # phi_a = 10/3, phi_b = 10/3, phi_c = 7/3  (sums to 9)
    assert math.isclose(phi["a"], 10 / 3, abs_tol=1e-6)
    assert math.isclose(phi["b"], 10 / 3, abs_tol=1e-6)
    assert math.isclose(phi["c"], 7 / 3, abs_tol=1e-6)


def test_pairwise_complementarity_matches_definition():
    players, v = _toy_game()
    comp = pairwise_complementarity(players, v)
    assert math.isclose(comp[("a", "b")], v[("a", "b")] - v[("a",)] - v[("b",)])
    assert math.isclose(comp[("a", "b")], 2.0)
    assert math.isclose(comp[("a", "c")], 0.0)
