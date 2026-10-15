from src.characteristic_function import all_coalitions, coalition_key, compute_characteristic_function
from src.network import load_all


def test_all_coalitions_count_and_includes_empty():
    players = ["a", "b", "c"]
    coalitions = all_coalitions(players)
    assert len(coalitions) == 2 ** len(players)
    assert () in coalitions
    assert tuple(players) in coalitions


def test_coalition_key_normalises_order():
    assert coalition_key(["b", "a"]) == coalition_key(["a", "b"]) == ("a", "b")


def test_characteristic_function_basic_properties():
    segments, services, od_df = load_all()
    players = list(services.keys())
    v, results = compute_characteristic_function(services, segments, od_df)

    # v(emptyset) = 0 by convention
    assert v[()] == 0.0

    # every coalition has a recorded value, 2^n total
    assert len(v) == 2 ** len(players)

    # grand coalition should be weakly the best achievable value across all
    # coalitions (super-additivity is NOT guaranteed in general games, but
    # here the grand coalition can always replicate any smaller coalition's
    # plan by leaving extra services at frequency 0, so it must be >= max)
    grand_value = v[coalition_key(players)]
    assert grand_value >= max(v.values()) - 1e-6


def test_grand_coalition_dominates_all_singletons():
    segments, services, od_df = load_all()
    players = list(services.keys())
    v, _ = compute_characteristic_function(services, segments, od_df)
    grand_value = v[coalition_key(players)]
    for p in players:
        assert grand_value >= v[coalition_key([p])] - 1e-6
