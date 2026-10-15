"""Shapley values, marginal contributions and pairwise complementarities
for the railway service cooperative game.

Shapley value (closed-form subset sum, exact for the small n=5 game here):

    phi_i = sum_{S subseteq N \\ {i}}  w(|S|, n) * [ v(S u {i}) - v(S) ]

    w(k, n) = k! (n-k-1)! / n!

where N is the set of all services and n = |N|. This is mathematically
identical to averaging the marginal contribution of i over all n!
permutations of N, but is computed directly from the already-enumerated
characteristic function (2^n terms) instead of generating permutations.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Tuple

from src.characteristic_function import coalition_key


def shapley_values(players: List[str], v: Dict[Tuple[str, ...], float]) -> Dict[str, float]:
    n = len(players)
    phi = {i: 0.0 for i in players}

    for i in players:
        others = [p for p in players if p != i]
        for k in range(len(others) + 1):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for subset in combinations(others, k):
                s_key = coalition_key(subset)
                si_key = coalition_key(list(subset) + [i])
                marginal = v[si_key] - v[s_key]
                phi[i] += weight * marginal

    return phi


def marginal_contributions_along_order(
    order: List[str], v: Dict[Tuple[str, ...], float]
) -> Dict[str, float]:
    """Marginal contribution of each player when added in a fixed order
    (used for the 'independent/greedy' narrative, not for phi itself)."""
    contributions = {}
    coalition: List[str] = []
    prev_value = v[coalition_key(coalition)]
    for player in order:
        coalition.append(player)
        new_value = v[coalition_key(coalition)]
        contributions[player] = new_value - prev_value
        prev_value = new_value
    return contributions


def pairwise_complementarity(players: List[str], v: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, str], float]:
    """c_ij = v({i,j}) - v({i}) - v({j}).

    c_ij > 0 means services i and j are complements (cooperating creates
    extra system benefit beyond what each achieves alone, e.g. by jointly
    covering demand neither can serve profitably by itself or by relieving
    each other's capacity bottlenecks). c_ij < 0 means they are substitutes
    / cannibalise each other's demand.
    """
    comp: Dict[Tuple[str, str], float] = {}
    for i, j in combinations(players, 2):
        v_i = v[coalition_key([i])]
        v_j = v[coalition_key([j])]
        v_ij = v[coalition_key([i, j])]
        comp[(i, j)] = v_ij - v_i - v_j
    return comp


def verify_efficiency(phi: Dict[str, float], v_grand: float, tol: float = 1e-3) -> bool:
    """Efficiency axiom: sum_i phi_i == v(N)."""
    return abs(sum(phi.values()) - v_grand) <= tol
