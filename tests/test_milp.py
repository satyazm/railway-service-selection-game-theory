import math

from src.milp import solve_service_selection_milp
from src.network import load_all


def test_grand_coalition_solves_optimal():
    segments, services, od_df = load_all()
    res = solve_service_selection_milp(services, segments, od_df, coalition=None)
    assert res.status == "Optimal"
    assert res.objective >= 0


def test_excluded_services_have_zero_frequency_and_flow():
    segments, services, od_df = load_all()
    coalition = ["Shatabdi_12027"]
    res = solve_service_selection_milp(services, segments, od_df, coalition=coalition)
    for s, f in res.frequencies.items():
        if s not in coalition:
            assert f == 0
    for (s, o, d), flow in res.flows.items():
        if s not in coalition:
            assert math.isclose(flow, 0.0, abs_tol=1e-6)


def test_capacity_constraints_respected():
    segments, services, od_df = load_all()
    res = solve_service_selection_milp(services, segments, od_df, coalition=None)
    for (s, seg_idx), load in res.segment_loads.items():
        cap_total = services[s].capacity * res.frequencies[s]
        assert load <= cap_total + 1e-6


def test_demand_constraints_respected():
    segments, services, od_df = load_all()
    res = solve_service_selection_milp(services, segments, od_df, coalition=None)
    demand = {(row.origin, row.destination): row.demand for row in od_df.itertuples()}
    totals = {}
    for (s, o, d), flow in res.flows.items():
        totals[(o, d)] = totals.get((o, d), 0.0) + flow
    for od, total in totals.items():
        assert total <= demand[od] + 1e-6


def test_frequency_bounds_respected():
    segments, services, od_df = load_all()
    res = solve_service_selection_milp(services, segments, od_df, coalition=None)
    for s, f in res.frequencies.items():
        assert 0 <= f <= services[s].max_freq


def test_empty_coalition_yields_zero_objective():
    segments, services, od_df = load_all()
    res = solve_service_selection_milp(services, segments, od_df, coalition=[])
    assert math.isclose(res.objective, 0.0, abs_tol=1e-6)
    assert res.active_services == []


def test_fixed_frequencies_are_honoured():
    segments, services, od_df = load_all()
    fixed = {"Shatabdi_12027": 2, "Lalbagh_12607": 1}
    res = solve_service_selection_milp(segments=segments, services=services, od_df=od_df,
                                        fixed_frequencies=fixed)
    assert res.frequencies["Shatabdi_12027"] == 2
    assert res.frequencies["Lalbagh_12607"] == 1
    for s in services:
        if s not in fixed:
            assert res.frequencies[s] == 0
