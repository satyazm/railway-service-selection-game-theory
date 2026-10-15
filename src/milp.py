"""MILP formulation for service selection, frequency setting and O-D seat
allocation on the synthetic railway network.

Mathematical formulation
-------------------------
Sets
    S       candidate services (restricted to a coalition when solving the
            cooperative game; services outside the coalition are forced to
            zero frequency)
    OD      origin-destination pairs with positive demand
    E       physical track segments

Decision variables
    f_s  >= 0, integer           frequency (trains/day) of service s in {0,...,maxFreq_s}
    y_{s,od} >= 0, continuous    passengers of OD pair od carried on service s
                                 (only defined for (s, od) pairs the service
                                 can physically serve)

Parameters
    cap_s        seats per train of service s
    cost_s       operating cost per train (frequency unit) of service s
    fare_{s,od}  fare per passenger of OD pair od when carried on service s,
                 = fare_per_km_s * distance_od (service-specific, since real
                 Indian Railways fares differ by class/train category for
                 the same physical trip -- see network.py)
    dem_od       daily demand for OD pair od
    A            0/1 segment-incidence matrix, L = A x  (see network.py),
                 mapping the stacked y vector (x) to per-(service,segment)
                 loads (L)

Objective (maximise net system benefit)
    max  sum_{s,od} fare_{s,od} * y_{s,od}  -  sum_s cost_s * f_s

Constraints
    (1) Demand:        sum_{s servable} y_{s,od} <= dem_od               for all od
    (2) Capacity:       L_{s,e} = sum_{od covering e on s} y_{s,od} <= cap_s * f_s
                                                                           for all (s,e) in route(s)
    (3) Coalition:      f_s = 0                           for all s not in coalition S
    (4) Bounds:         0 <= f_s <= maxFreq_s, integer;    y_{s,od} >= 0

This is exactly the L = A x device from network.py: constraint (2) is
``A @ y <= cap ⊗ f`` once y is stacked in the column order network.py
defines, so the capacity constraints are literally read off the rows of A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

from src.network import Service, Segment, build_assignment_matrix


@dataclass
class MILPResult:
    status: str
    objective: float
    frequencies: Dict[str, int]
    flows: Dict[Tuple[str, str, str], float]      # (service, o, d) -> passengers
    segment_loads: Dict[Tuple[str, int], float]   # (service, segment_index) -> load
    revenue: float
    operating_cost: float
    active_services: List[str]


def solve_service_selection_milp(
    services: Dict[str, Service],
    segments: List[Segment],
    od_df: pd.DataFrame,
    coalition: Optional[Iterable[str]] = None,
    fixed_frequencies: Optional[Dict[str, int]] = None,
    solver_msg: bool = False,
) -> MILPResult:
    """Solve the service-selection / seat-allocation MILP.

    Parameters
    ----------
    coalition : iterable of service names allowed to operate (frequency may
        be > 0). ``None`` means the grand coalition (all services allowed).
        Services not in the coalition are fixed at frequency 0, which also
        forces every y_{s,od} for that service to 0 via the capacity
        constraint (capacity becomes 0). Ignored when ``fixed_frequencies``
        is given.
    fixed_frequencies : optional mapping service -> frequency. When given,
        every service's frequency is pinned to this value (0 for services
        omitted from the mapping) instead of being decided jointly, and the
        model collapses to a pure LP that only re-optimises the O-D seat
        allocation y given those fixed, uncoordinated frequencies. This is
        used to model "independent" service planning, where each operator
        sets its own frequency in isolation and only the passengers'
        allocation across already-running trains is still solved for.
    """
    all_service_names = list(services.keys())
    coalition_set = set(all_service_names) if coalition is None else set(coalition)

    demand = {(row.origin, row.destination): row.demand for row in od_df.itertuples()}
    distance = {(row.origin, row.destination): row.distance_km for row in od_df.itertuples()}

    A, rows, columns = build_assignment_matrix(services, segments)

    prob = pulp.LpProblem("railway_service_selection", pulp.LpMaximize)

    freq_vars = {
        s: pulp.LpVariable(f"f_{s}", lowBound=0, upBound=services[s].max_freq, cat="Integer")
        for s in all_service_names
    }
    y_vars = {
        (s, o, d): pulp.LpVariable(f"y_{s}_{o}_{d}", lowBound=0, cat="Continuous")
        for (s, o, d) in columns
    }

    # Objective: revenue - operating cost. Fare is service-specific
    # (fare_per_km of the operating service times the OD distance), since
    # real Indian Railways fares differ by train/class for the same trip.
    fare = {(s, o, d): services[s].fare_per_km * distance[(o, d)] for (s, o, d) in columns}
    revenue_terms = [fare[(s, o, d)] * y_vars[(s, o, d)] for (s, o, d) in columns]
    cost_terms = [services[s].cost_per_freq * freq_vars[s] for s in all_service_names]
    prob += pulp.lpSum(revenue_terms) - pulp.lpSum(cost_terms)

    if fixed_frequencies is not None:
        # Independent/uncoordinated planning: frequencies are pinned, only
        # seat allocation is still optimised.
        for s in all_service_names:
            prob += freq_vars[s] == int(fixed_frequencies.get(s, 0)), f"fixed_freq_{s}"
    else:
        # Coalition constraint: non-member services cannot run.
        for s in all_service_names:
            if s not in coalition_set:
                prob += freq_vars[s] == 0, f"coalition_excl_{s}"

    # Demand constraints.
    for (o, d) in demand:
        relevant = [y_vars[(s, oo, dd)] for (s, oo, dd) in columns if (oo, dd) == (o, d)]
        if relevant:
            prob += pulp.lpSum(relevant) <= demand[(o, d)], f"demand_{o}_{d}"

    # Capacity constraints, one per (service, segment) row of A, read
    # directly off the assignment matrix: L_{s,e} <= cap_s * f_s.
    col_vars = [y_vars[c] for c in columns]
    for row_idx, (s, seg_idx) in enumerate(rows):
        load_expr = pulp.lpSum(
            A[row_idx, j] * col_vars[j] for j in range(len(columns)) if A[row_idx, j] != 0
        )
        prob += load_expr <= services[s].capacity * freq_vars[s], f"cap_{s}_{seg_idx}"

    solver = pulp.PULP_CBC_CMD(msg=solver_msg)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    frequencies = {s: int(round(freq_vars[s].value() or 0)) for s in all_service_names}
    flows = {(s, o, d): float(y_vars[(s, o, d)].value() or 0.0) for (s, o, d) in columns}

    segment_loads: Dict[Tuple[str, int], float] = {}
    for row_idx, (s, seg_idx) in enumerate(rows):
        load = sum(A[row_idx, j] * flows[columns[j]] for j in range(len(columns)))
        segment_loads[(s, seg_idx)] = load

    revenue = sum(fare[(s, o, d)] * flows[(s, o, d)] for (s, o, d) in columns)
    operating_cost = sum(services[s].cost_per_freq * frequencies[s] for s in all_service_names)
    objective = pulp.value(prob.objective) if prob.objective is not None else 0.0
    active_services = [s for s in all_service_names if frequencies[s] > 0]

    return MILPResult(
        status=status,
        objective=float(objective),
        frequencies=frequencies,
        flows=flows,
        segment_loads=segment_loads,
        revenue=float(revenue),
        operating_cost=float(operating_cost),
        active_services=active_services,
    )


@dataclass
class LPDualResult:
    status: str
    revenue: float
    flows: Dict[Tuple[str, str, str], float]
    demand_duals: Dict[Tuple[str, str], float]       # shadow price per unit of OD demand
    capacity_duals: Dict[Tuple[str, int], float]     # shadow price per unit of (service,segment) capacity


def solve_lp_with_duals(
    services: Dict[str, Service],
    segments: List[Segment],
    od_df: pd.DataFrame,
    frequencies: Dict[str, int],
    solver_msg: bool = False,
) -> LPDualResult:
    """Pure LP (no integer/decision variables for frequency at all -- every
    service's frequency is baked in as a plain number) that only solves the
    O-D seat allocation y, given fixed capacities ``cap_s * frequencies[s]``.

    Dual values (shadow prices) are only well-defined for a true LP; CBC
    does not report meaningful duals for a problem that still contains
    integer variables (even ones pinned via equality constraints), which is
    why this is a separate function from ``solve_service_selection_milp``
    rather than reusing its ``fixed_frequencies`` mode.

    The shadow prices returned here are exactly the y^*, lambda^*, mu^* of
    LP duality used for the Owen (1975) "linear production game" style
    allocation in ``src/core_analysis.py``: ``capacity_duals`` is the
    marginal value of one more seat of capacity on a service's own route
    segment (a resource only that service privately owns), and
    ``demand_duals`` is the marginal value of one more unit of a given O-D
    pair's demand (a resource shared/competed for by every service able to
    carry that O-D pair).
    """
    demand = {(row.origin, row.destination): row.demand for row in od_df.itertuples()}
    distance = {(row.origin, row.destination): row.distance_km for row in od_df.itertuples()}

    A, rows, columns = build_assignment_matrix(services, segments)

    prob = pulp.LpProblem("railway_seat_allocation_lp", pulp.LpMaximize)
    y_vars = {
        (s, o, d): pulp.LpVariable(f"y_{s}_{o}_{d}", lowBound=0, cat="Continuous")
        for (s, o, d) in columns
    }

    fare = {(s, o, d): services[s].fare_per_km * distance[(o, d)] for (s, o, d) in columns}
    prob += pulp.lpSum(fare[c] * y_vars[c] for c in columns)

    demand_constraint_names: Dict[Tuple[str, str], str] = {}
    for (o, d) in demand:
        relevant = [y_vars[(s, oo, dd)] for (s, oo, dd) in columns if (oo, dd) == (o, d)]
        if relevant:
            cname = f"demand_{o}_{d}"
            prob += pulp.lpSum(relevant) <= demand[(o, d)], cname
            demand_constraint_names[(o, d)] = cname

    col_vars = [y_vars[c] for c in columns]
    capacity_constraint_names: Dict[Tuple[str, int], str] = {}
    for row_idx, (s, seg_idx) in enumerate(rows):
        load_expr = pulp.lpSum(
            A[row_idx, j] * col_vars[j] for j in range(len(columns)) if A[row_idx, j] != 0
        )
        cname = f"cap_{s}_{seg_idx}"
        prob += load_expr <= services[s].capacity * frequencies.get(s, 0), cname
        capacity_constraint_names[(s, seg_idx)] = cname

    solver = pulp.PULP_CBC_CMD(msg=solver_msg)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    flows = {c: float(y_vars[c].value() or 0.0) for c in columns}
    revenue = sum(fare[c] * flows[c] for c in columns)

    demand_duals = {od: (prob.constraints[cname].pi or 0.0) for od, cname in demand_constraint_names.items()}
    capacity_duals = {key: (prob.constraints[cname].pi or 0.0) for key, cname in capacity_constraint_names.items()}

    return LPDualResult(
        status=status,
        revenue=float(revenue),
        flows=flows,
        demand_duals=demand_duals,
        capacity_duals=capacity_duals,
    )
