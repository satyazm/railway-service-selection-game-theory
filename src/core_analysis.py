"""Core stability and LP-duality allocation for the railway cooperative
game.

Two separate questions, both raised by the guide's notes:

1. **Is the grand coalition stable?** The Shapley value is a *fair* way to
   split v(N), but fairness does not imply *stability*: if some sub-
   coalition S could do strictly better by seceding and keeping only
   v(S) for itself, the allocation (and the whole arrangement of pooling
   all services under one system) is not self-enforcing. The **core** is
   the set of allocations x (with sum_i x_i = v(N)) such that no
   coalition can do better alone: sum_{i in S} x_i >= v(S) for every
   S subset N. If the core is empty, no allocation is simultaneously fair
   *and* stable -- an important, genuine finding to report, not a bug.

   We check this with the standard **least-core LP** (Maschler, Peleg &
   Shapley 1979): minimise the smallest uniform relaxation epsilon of
   every coalition's rationality constraint needed to restore
   feasibility. ``epsilon* <= 0`` iff the (exact) core is non-empty, and
   the minimiser is itself a core allocation in that case.

        min  epsilon
        s.t. sum_i x_i = v(N)
             sum_{i in S} x_i >= v(S) - epsilon      for every proper,
                                                       non-empty S subset N

2. **A second, LP-duality-based allocation.** For games defined by a
   linear program (Owen, 1975, "On the core of linear production games"),
   the dual prices of the grand coalition's optimal LP give an allocation
   that is provably in the core whenever each player's own private
   resource rows are what is being priced. Here, each service's own
   capacity (cap_s * f_s) is a resource only that service contributes --
   no other service can use service s's trains -- so the dual (shadow)
   price of service s's own capacity constraints, evaluated at its actual
   capacity, is a principled "resource rent" for service s.

   This decomposes the optimal LP revenue, by strong duality, as

        revenue* = sum_s (resource rent of s's own capacity)
                   + sum_od (demand-scarcity rent of OD pair od)

   The second term is *not* privately owned by any single service -- it is
   the value created by O-D demand being scarce relative to pooled
   capacity, which only exists because services compete/cooperate on
   shared passenger demand. We report it explicitly as the "pooled-demand
   rent" and, to get a single comparable per-service number, redistribute
   it pro rata to each service's share of realised revenue (a transparent,
   documented choice -- not claimed to be the unique "correct" split).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import pulp

from src.characteristic_function import all_coalitions, coalition_key
from src.milp import MILPResult, solve_lp_with_duals
from src.network import Segment, Service


def least_core(players: List[str], v: Dict[Tuple[str, ...], float]) -> Dict:
    """Solve the least-core LP. Returns epsilon, whether the exact core is
    non-empty (epsilon <= tol), and the minimising (least-core) allocation.
    """
    n = len(players)
    grand_key = coalition_key(players)

    prob = pulp.LpProblem("least_core", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}") for i in players}  # free (can be negative)
    epsilon = pulp.LpVariable("epsilon")
    prob += epsilon

    prob += pulp.lpSum(x[i] for i in players) == v[grand_key], "efficiency"

    for S in all_coalitions(players):
        if len(S) == 0 or len(S) == n:
            continue
        key = coalition_key(S)
        prob += pulp.lpSum(x[i] for i in S) >= v[key] - epsilon, f"rationality_{'_'.join(key)}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    allocation = {i: float(x[i].value() or 0.0) for i in players}
    eps_value = float(epsilon.value() or 0.0)

    return {
        "status": pulp.LpStatus[prob.status],
        "epsilon": eps_value,
        "in_core": eps_value <= 1e-6,
        "allocation": allocation,
    }


def core_violations(
    players: List[str], v: Dict[Tuple[str, ...], float], allocation: Dict[str, float], tol: float = 1e-6
) -> List[Tuple[Tuple[str, ...], float]]:
    """Coalitions that would strictly prefer to secede given `allocation`.
    Returns a list of (coalition, shortfall) where shortfall = v(S) - sum(x_i for i in S) > 0.
    """
    violations = []
    n = len(players)
    for S in all_coalitions(players):
        if len(S) == 0 or len(S) == n:
            continue
        key = coalition_key(S)
        lhs = sum(allocation[i] for i in S)
        shortfall = v[key] - lhs
        if shortfall > tol:
            violations.append((key, shortfall))
    return sorted(violations, key=lambda t: -t[1])


def lp_dual_allocation(
    services: Dict[str, Service],
    segments: List[Segment],
    od_df: pd.DataFrame,
    grand_result: MILPResult,
) -> Dict:
    """LP-duality-based allocation at the grand coalition's optimum (see
    module docstring). Frequencies are fixed at the grand coalition's
    MILP-optimal values, the seat-allocation LP is re-solved for its dual
    prices, and each service's "resource rent" is the shadow value of its
    own privately-owned capacity minus its own operating cost. The
    remaining "pooled-demand rent" (not privately attributable to any one
    service) is distributed pro rata to each service's realised revenue
    share, to produce a single efficient (budget-balanced) allocation
    comparable to the Shapley value and the least-core allocation.
    """
    frequencies = grand_result.frequencies
    lp = solve_lp_with_duals(services, segments, od_df, frequencies)

    demand = {(row.origin, row.destination): row.demand for row in od_df.itertuples()}

    resource_rent: Dict[str, float] = {}
    for s, svc in services.items():
        capacity_value = sum(
            dual * (svc.capacity * frequencies.get(s, 0))
            for (ss, seg_idx), dual in lp.capacity_duals.items()
            if ss == s
        )
        cost = svc.cost_per_freq * frequencies.get(s, 0)
        resource_rent[s] = capacity_value - cost

    pooled_demand_rent = sum(dual * demand[od] for od, dual in lp.demand_duals.items())

    # Strong LP duality identity check: LP revenue == sum of capacity-value
    # + demand-value (both evaluated at the optimal dual prices).
    total_capacity_value = sum(
        dual * (services[s].capacity * frequencies.get(s, 0)) for (s, _), dual in lp.capacity_duals.items()
    )
    duality_gap = lp.revenue - (total_capacity_value + pooled_demand_rent)

    # realised revenue per service, to pro-rate the pooled-demand rent
    revenue_by_service: Dict[str, float] = {s: 0.0 for s in services}
    distance = {(row.origin, row.destination): row.distance_km for row in od_df.itertuples()}
    for (s, o, d), flow in lp.flows.items():
        revenue_by_service[s] += services[s].fare_per_km * distance[(o, d)] * flow
    total_revenue = sum(revenue_by_service.values())

    allocation: Dict[str, float] = {}
    for s in services:
        share = (revenue_by_service[s] / total_revenue) if total_revenue > 0 else 1.0 / len(services)
        allocation[s] = resource_rent[s] + share * pooled_demand_rent

    return {
        "resource_rent": resource_rent,
        "pooled_demand_rent": pooled_demand_rent,
        "duality_gap": duality_gap,
        "allocation": allocation,
        "total_allocated": sum(allocation.values()),
        "grand_coalition_value": grand_result.objective,
    }
