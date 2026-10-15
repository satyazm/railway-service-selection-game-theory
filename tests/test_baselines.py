from src.baselines import compare_strategies, global_optimum, greedy_selection, independent_selection
from src.characteristic_function import coalition_key, compute_characteristic_function
from src.network import load_all


def test_global_optimum_equals_grand_coalition_value():
    segments, services, od_df = load_all()
    players = list(services.keys())
    v, _ = compute_characteristic_function(services, segments, od_df)
    svcs, value = global_optimum(players, v)
    assert set(svcs) == set(players)
    assert value == v[coalition_key(players)]


def test_greedy_never_exceeds_global_optimum():
    segments, services, od_df = load_all()
    players = list(services.keys())
    v, _ = compute_characteristic_function(services, segments, od_df)
    _, greedy_value = greedy_selection(players, v)
    _, opt_value = global_optimum(players, v)
    assert greedy_value <= opt_value + 1e-6


def test_independent_never_exceeds_global_optimum():
    segments, services, od_df = load_all()
    players = list(services.keys())
    v, _ = compute_characteristic_function(services, segments, od_df)
    _, ind_value = independent_selection(players, v, services, segments, od_df)
    _, opt_value = global_optimum(players, v)
    assert ind_value <= opt_value + 1e-6


def test_compare_strategies_returns_all_three():
    segments, services, od_df = load_all()
    players = list(services.keys())
    v, _ = compute_characteristic_function(services, segments, od_df)
    comparison = compare_strategies(players, v, services, segments, od_df)
    assert set(comparison.keys()) == {"independent", "greedy", "global_optimal"}
    for entry in comparison.values():
        assert "services" in entry and "value" in entry
