"""Characteristic function v(S) for the railway service cooperative game.

v(S) is defined as the optimal objective value (maximum net system benefit:
fare revenue minus operating cost) of the service-selection MILP when only
the services in coalition S are allowed to operate (services outside S are
forced to zero frequency, per constraint (3) in milp.py). Services inside S
that are unprofitable to run are simply left at frequency 0 by the
optimizer -- the coalition is not forced to activate every member, which
matches the standard "coalition can always do at least as well as any of
its own subsets" property required of a characteristic function.

v(emptyset) = 0 by convention (no services, no revenue, no cost).
"""

from __future__ import annotations

from itertools import chain, combinations
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.milp import MILPResult, solve_service_selection_milp
from src.network import Segment, Service


def all_coalitions(players: List[str]) -> List[Tuple[str, ...]]:
    """All 2^n subsets of players, including the empty set, as tuples."""
    return list(chain.from_iterable(combinations(players, r) for r in range(len(players) + 1)))


def compute_characteristic_function(
    services: Dict[str, Service],
    segments: List[Segment],
    od_df: pd.DataFrame,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[str, ...], float], Dict[Tuple[str, ...], MILPResult]]:
    """Enumerate every coalition, solve its MILP, and record v(S).

    Returns
    -------
    values : dict mapping coalition (sorted tuple of service names) -> v(S)
    results : dict mapping coalition -> full MILPResult (for inspection)
    """
    players = list(services.keys())
    values: Dict[Tuple[str, ...], float] = {}
    results: Dict[Tuple[str, ...], MILPResult] = {}

    for coalition in all_coalitions(players):
        key = coalition_key(coalition)
        if len(coalition) == 0:
            values[key] = 0.0
            continue
        res = solve_service_selection_milp(services, segments, od_df, coalition=coalition)
        values[key] = res.objective
        results[key] = res
        if verbose:
            print(f"v{key} = {res.objective:.1f}  (active: {res.active_services})")

    return values, results


def coalition_key(services: Iterable) -> Tuple[str, ...]:
    """Normalise any iterable of service names into the canonical sorted-tuple key."""
    return tuple(sorted(services))
