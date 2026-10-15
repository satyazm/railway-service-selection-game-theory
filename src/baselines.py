"""Comparison of three service-selection strategies:

1. Independent selection (uncoordinated)
   Each service decides its own frequency by solving its own solo MILP
   (coalition = {i}), in isolation from everyone else, i.e. "my own train
   makes money on its own, so I'll run it at my own profit-maximising
   frequency." Those frequencies are then pinned and all such services run
   together, but -- crucially -- nobody re-optimises frequencies jointly
   afterwards, so overlapping services may both chase the same O-D demand
   without coordination. Only the resulting seat allocation is re-solved
   (via ``fixed_frequencies`` in milp.py) to find out how much net benefit
   this uncoordinated plan actually realises once the services interact.
   This exposes the cost of not cooperating, which is the entire point of
   contrasting it with the cooperative-game analysis.

2. Greedy coalition formation
   Start from the empty coalition and repeatedly add whichever remaining
   service yields the largest marginal contribution v(S u {i}) - v(S) in
   the fully-reoptimised joint MILP, stopping once no addition improves the
   value. A classic greedy heuristic for coalition structure formation.

3. Globally optimal (grand coalition)
   v(N): the full MILP solved once over all services, jointly choosing
   every frequency and every passenger's seat allocation. Because the MILP
   always has the option to set any service's frequency to 0, this is
   simultaneously the best *subset* and the best *allocation* -- i.e. the
   true social optimum that the other two strategies are judged against.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.characteristic_function import coalition_key
from src.milp import solve_service_selection_milp
from src.network import Segment, Service


def independent_selection(
    players: List[str],
    v: Dict[Tuple[str, ...], float],
    services: Dict[str, Service],
    segments: List[Segment],
    od_df: pd.DataFrame,
) -> Tuple[List[str], float]:
    # Each service independently decides to run only if profitable alone,
    # at its own solo-optimal frequency.
    solo_frequencies: Dict[str, int] = {}
    selected: List[str] = []
    for p in players:
        if v[coalition_key([p])] > 0:
            solo_result = solve_service_selection_milp(services, segments, od_df, coalition=[p])
            solo_frequencies[p] = solo_result.frequencies[p]
            selected.append(p)

    if not selected:
        return [], 0.0

    # Frequencies are pinned exactly as each service chose them in
    # isolation; only the O-D seat allocation is re-solved so passengers
    # can still pick among whichever trains are actually running.
    uncoordinated = solve_service_selection_milp(
        services, segments, od_df, fixed_frequencies=solo_frequencies
    )
    return selected, uncoordinated.objective


def greedy_selection(players: List[str], v: Dict[Tuple[str, ...], float]) -> Tuple[List[str], float]:
    coalition: List[str] = []
    current_value = v[coalition_key(coalition)]
    remaining = list(players)

    improved = True
    while improved and remaining:
        improved = False
        best_gain, best_player = -float("inf"), None
        for p in remaining:
            candidate_value = v[coalition_key(coalition + [p])]
            gain = candidate_value - current_value
            if gain > best_gain:
                best_gain, best_player = gain, p
        if best_player is not None and best_gain > 0:
            coalition.append(best_player)
            remaining.remove(best_player)
            current_value = v[coalition_key(coalition)]
            improved = True

    return coalition, current_value


def global_optimum(players: List[str], v: Dict[Tuple[str, ...], float]) -> Tuple[List[str], float]:
    grand = coalition_key(players)
    return list(players), v[grand]


def compare_strategies(
    players: List[str],
    v: Dict[Tuple[str, ...], float],
    services: Dict[str, Service],
    segments: List[Segment],
    od_df: pd.DataFrame,
) -> Dict[str, Dict]:
    ind_services, ind_value = independent_selection(players, v, services, segments, od_df)
    greedy_services, greedy_value = greedy_selection(players, v)
    opt_services, opt_value = global_optimum(players, v)

    return {
        "independent": {"services": ind_services, "value": ind_value},
        "greedy": {"services": greedy_services, "value": greedy_value},
        "global_optimal": {"services": opt_services, "value": opt_value},
    }
