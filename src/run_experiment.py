"""End-to-end experiment driver.

Run with::

    python -m src.run_experiment

from the project root (with the virtualenv activated). This regenerates the
synthetic CSVs, solves the grand-coalition MILP, enumerates all 32
coalitions, computes Shapley values and pairwise complementarities, runs
the independent/greedy/optimal comparison, saves all figures to
results/figures/, and writes a results summary to results/.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from src.baselines import compare_strategies
from src.characteristic_function import compute_characteristic_function, coalition_key
from src.core_analysis import core_violations, least_core, lp_dual_allocation
from src.milp import solve_service_selection_milp
from src.network import load_all, write_csvs
from src.shapley import pairwise_complementarity, shapley_values, verify_efficiency
from src.visualize import (
    plot_allocation_comparison,
    plot_coalition_values,
    plot_network,
    plot_pairwise_complementarity,
    plot_segment_utilization,
    plot_shapley_values,
    plot_strategy_comparison,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("1. Generating synthetic network, demand and service CSVs ...")
    write_csvs()
    segments, services, od_df = load_all()
    players = list(services.keys())

    print("2. Solving the grand-coalition MILP (globally optimal plan) ...")
    grand_result = solve_service_selection_milp(services, segments, od_df, coalition=None)
    print(f"   status={grand_result.status}  objective={grand_result.objective:.2f}  "
          f"active_services={grand_result.active_services}")

    print("3. Enumerating all 2^5 = 32 coalitions and solving their MILPs ...")
    v, coalition_results = compute_characteristic_function(services, segments, od_df, verbose=True)

    print("4. Computing Shapley values ...")
    phi = shapley_values(players, v)
    grand_value = v[coalition_key(players)]
    efficient = verify_efficiency(phi, grand_value)
    print(f"   Shapley values: {phi}")
    print(f"   Efficiency check (sum(phi) == v(N)): {efficient}")

    print("5. Computing pairwise complementarities ...")
    comp = pairwise_complementarity(players, v)

    print("6. Comparing independent / greedy / globally-optimal strategies ...")
    comparison = compare_strategies(players, v, services, segments, od_df)
    print(json.dumps(comparison, indent=2))

    print("7. Checking core stability (least-core LP) ...")
    lc = least_core(players, v)
    lc_violations = core_violations(players, v, lc["allocation"])
    print(f"   least-core epsilon={lc['epsilon']:.2f}  in_core={lc['in_core']}  "
          f"({len(lc_violations)} coalitions would still prefer to secede)")

    print("8. Computing LP-duality-based allocation ...")
    dual = lp_dual_allocation(services, segments, od_df, grand_result)
    dual_violations = core_violations(players, v, dual["allocation"])
    print(f"   resource_rent={dual['resource_rent']}")
    print(f"   pooled_demand_rent={dual['pooled_demand_rent']:.2f}  duality_gap={dual['duality_gap']:.4f}")
    print(f"   ({len(dual_violations)} coalitions would still prefer to secede)")

    shapley_violations = core_violations(players, v, phi)
    print(f"   (Shapley value: {len(shapley_violations)} coalitions would still prefer to secede)")

    print("9. Rendering visualizations ...")
    fig_paths = {
        "network": plot_network(services, segments),
        "segment_utilization": plot_segment_utilization(grand_result, services, segments),
        "coalition_values": plot_coalition_values(v),
        "shapley_values": plot_shapley_values(phi),
        "pairwise_complementarity": plot_pairwise_complementarity(players, comp),
        "strategy_comparison": plot_strategy_comparison(comparison),
        "allocation_comparison": plot_allocation_comparison(players, phi, lc["allocation"], dual["allocation"]),
    }
    for name, path in fig_paths.items():
        print(f"   saved {name} -> {path}")

    print("10. Writing results summary ...")
    coalition_table = pd.DataFrame(
        [{"coalition": "{" + ",".join(c) + "}" if c else "{}", "num_services": len(c), "value": val}
         for c, val in sorted(v.items(), key=lambda kv: (len(kv[0]), kv[0]))]
    )
    coalition_table.to_csv(os.path.join(RESULTS_DIR, "coalition_values.csv"), index=False)

    shapley_table = pd.DataFrame({"service": list(phi.keys()), "shapley_value": list(phi.values())})
    shapley_table.to_csv(os.path.join(RESULTS_DIR, "shapley_values.csv"), index=False)

    comp_table = pd.DataFrame(
        [{"service_i": i, "service_j": j, "complementarity": val} for (i, j), val in comp.items()]
    )
    comp_table.to_csv(os.path.join(RESULTS_DIR, "pairwise_complementarity.csv"), index=False)

    allocation_table = pd.DataFrame({
        "service": players,
        "shapley_value": [phi[p] for p in players],
        "least_core_allocation": [lc["allocation"][p] for p in players],
        "lp_dual_allocation": [dual["allocation"][p] for p in players],
        "lp_dual_resource_rent": [dual["resource_rent"][p] for p in players],
    })
    allocation_table.to_csv(os.path.join(RESULTS_DIR, "allocation_comparison.csv"), index=False)

    summary = {
        "grand_coalition_value": grand_value,
        "grand_coalition_active_services": grand_result.active_services,
        "shapley_efficiency_holds": efficient,
        "strategy_comparison": comparison,
        "core_analysis": {
            "least_core_epsilon": lc["epsilon"],
            "exact_core_is_empty": not lc["in_core"],
            "shapley_core_violations": len(shapley_violations),
            "least_core_allocation_violations": len(lc_violations),
            "lp_dual_allocation_violations": len(dual_violations),
            "lp_dual_pooled_demand_rent": dual["pooled_demand_rent"],
            "lp_dual_duality_gap": dual["duality_gap"],
        },
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
