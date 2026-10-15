"""Real railway network, demand model and candidate service construction.

Real data used
--------------
The station list, station coordinates, train identities/routes/distances
and the actual stop sequences below come from the Kaggle dataset
"Indian Railways Dataset" (sripaadsrinivasan/indian-railways-dataset,
CC0 Public Domain), itself sourced from Indian Railways' public NTES/time-
table data. A curated extract of that dataset -- just the records needed
for this corridor -- is checked in at ``data/real_corridor_source.json``
(the full raw files, ~98 MB, are not kept in this repo; see
"Reproducing the raw download" below to regenerate the extract).

Corridor: the real Chennai Central -- Bangalore City mainline
    MAS (Chennai Central) -- AJJ (Arakkonam) -- KPD (Katpadi Jn)
    -- JTJ (Jolarpettai) -- BWT (Bangarapet) -- KJM (Krishnarajapuram)
    -- SBC (Bangalore City Jn)
7 real stations, confirmed to be the actual physical station order on this
line (both by their real geographic coordinates and by the real stop
sequence of the 5 trains below, as recorded in schedules.json).

Candidate services: 5 real, named Indian Railways trains that actually
operate on this corridor, with their real train numbers, real origin/
destination, real total distance and real class composition (which
stop sequence of the 7 corridor stations is reconstructed per train from
the real schedule data):

    12027  Chennai Central -- Bangalore City Shatabdi Express  (all 7 stops)
    12607  Chennai Central -- Bangalore  Lalbagh Express        (all 7 stops)
    12657  Chennai -- Bangalore Mail                             (all 7 stops)
    56261  Arakkonam -- Bangalore Passenger               (AJJ..SBC, 6 stops)
    43401  Chennai Central -- Arakkonam Fast (MEMU)         (MAS..AJJ, 2 stops)

What is real vs. modelled
--------------------------
REAL: station identities, coordinates, physical corridor order, segment
distances (haversine great-circle distance between real coordinates),
train identities/numbers/names, which of the 7 corridor stations each
train actually stops at, and each train's broad class composition
(whether it carries 1A/2A/3A/Sleeper/Chair-Car coaches).

MODELLED (clearly separated, not claimed as live/confidential data):
- **Seating capacity per train** and **operating cost per train-km**: the
  dataset only records *which* classes a train carries (as 0/1 flags), not
  real coach counts, and Indian Railways does not publish per-train
  operating cost. These are derived from standard, publicly documented
  typical Indian Railways rake compositions and representative train-km
  cost ranges (RITES/industry order-of-magnitude figures), looked up by
  train category -- see CAPACITY_BY_CATEGORY / COST_PER_KM_BY_CATEGORY
  below. They are explicitly representative assumptions, not measured
  values.
- **Fare per passenger-km**: representative IRCTC base-fare-per-class
  slabs (public, approximate), looked up by train category -- see
  FARE_PER_KM_BY_CATEGORY below.
- **O-D passenger demand**: Indian Railways does not publish station-level
  ticketing O-D matrices. Demand here is synthesised with a standard
  gravity model, `demand_ij = k * pop_i * pop_j / dist_ij^2`, calibrated
  with the **real, approximate 2011-census population** of the towns/
  cities each station actually serves (see STATION_POPULATION). This is a
  standard, transparent technique in transport-planning research for
  estimating demand where ticketing microdata is restricted, and is
  reported here as exactly that -- a population-calibrated estimate, not
  observed ridership.

Reproducing the raw download
-----------------------------
```
# via the Kaggle API/CLI (requires a free Kaggle account + API token)
kaggle datasets download -d sripaadsrinivasan/indian-railways-dataset
# then filter stations.json / trains.json / schedules.json to the station
# codes MAS, AJJ, KPD, JTJ, BWT, KJM, SBC and train numbers 12027, 12607,
# 12657, 56261, 43401 to reproduce data/real_corridor_source.json
```
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REAL_SOURCE_PATH = os.path.join(DATA_DIR, "real_corridor_source.json")

RNG_SEED = 42

# Real physical order of the 7 corridor stations (confirmed by both real
# coordinates and the real recorded stop sequences of the trains below).
STATIONS: List[str] = ["MAS", "AJJ", "KPD", "JTJ", "BWT", "KJM", "SBC"]

# Real, approximate town/city population served by each station (2011
# Census of India, public, approximate -- Arakkonam/Jolarpettai/Bangarapet/
# Krishnarajapuram figures are representative of the local town/suburb,
# not the wider district). Used only to calibrate synthetic O-D demand.
STATION_POPULATION: Dict[str, int] = {
    "MAS": 7_088_000,   # Chennai city proper
    "AJJ": 74_000,       # Arakkonam town
    "KPD": 504_000,      # Vellore city (Katpadi is Vellore's railway gateway)
    "JTJ": 20_000,       # Jolarpettai town
    "BWT": 35_000,       # Bangarapet town
    "KJM": 60_000,       # Krishnarajapuram's own local catchment (a station
                          # within the Bangalore agglomeration -- deliberately
                          # NOT the whole metro population, which would
                          # double-count SBC's own 8.44M and make every
                          # KJM pair dominate the gravity model)
    "SBC": 8_443_675,    # Bangalore (Bengaluru) city proper
}

# Real train number -> (display name, service "category"). Category drives
# the modelled capacity/cost/fare lookups below.
REAL_TRAINS: Dict[str, Dict[str, str]] = {
    "12027": {"display": "Shatabdi_12027", "category": "Shatabdi"},
    "12607": {"display": "Lalbagh_12607", "category": "SF_ChairCar"},
    "12657": {"display": "Mail_12657", "category": "Mail_Mixed"},
    "56261": {"display": "Passenger_56261", "category": "Passenger"},
    "43401": {"display": "MEMU_43401", "category": "MEMU"},
}

# ---------------------------------------------------------------------
# Modelled parameters (documented assumptions -- see module docstring).
# ---------------------------------------------------------------------

# Seats per train, by service category. Representative of a typical rake
# for that category of Indian train (not the literal coach count flags in
# the source dataset, which are presence/absence booleans, not counts).
CAPACITY_BY_CATEGORY: Dict[str, float] = {
    "Shatabdi": 850,      # ~2 Executive Chair + ~10 Chair Car coaches
    "SF_ChairCar": 950,   # ~12-13 Chair Car coaches (Lalbagh/Brindavan-style)
    "Mail_Mixed": 1100,   # mixed 1A/2A/3A/Sleeper rake, ~20 coaches
    "Passenger": 1200,    # unreserved/general rake, ~14-16 coaches
    "MEMU": 1000,         # dense suburban-style seating, ~12 coaches
}

# Representative operating cost per train-km, by category (order-of-
# magnitude public/industry figures; premium AC rakes cost more per km to
# run than unreserved passenger/MEMU rakes).
COST_PER_KM_BY_CATEGORY: Dict[str, float] = {
    "Shatabdi": 120.0,
    "SF_ChairCar": 90.0,
    "Mail_Mixed": 100.0,
    "Passenger": 40.0,
    "MEMU": 35.0,
}

# Representative blended fare per passenger-km, by category (approximate
# public IRCTC base-fare-per-class slabs, blended across the class mix
# each category typically carries).
FARE_PER_KM_BY_CATEGORY: Dict[str, float] = {
    "Shatabdi": 1.8,
    "SF_ChairCar": 1.2,
    "Mail_Mixed": 0.8,
    "Passenger": 0.30,
    "MEMU": 0.25,
}

# Maximum number of daily train-slots that could realistically be operated
# on this stopping pattern / class-mix (service-network-design sense: a
# "service" is a stopping pattern + class mix, which in practice could be
# replicated across several daily train paths, not literally one single
# named train). Higher for cheaper/short-haul/commuter patterns.
MAX_FREQ_BY_CATEGORY: Dict[str, int] = {
    "Shatabdi": 3,
    "SF_ChairCar": 4,
    "Mail_Mixed": 3,
    "Passenger": 6,
    "MEMU": 10,
}

# Gravity-model calibration for the synthetic O-D demand (see module
# docstring): demand_ij = K * sqrt(pop_i * pop_j) / max(dist_ij, FLOOR_KM).
# The square root on the population product (rather than a raw product)
# avoids the classic gravity-model blow-up where the two biggest cities
# swamp every other pair; the distance floor represents a fixed minimum
# trip "friction" (station access/egress, interchange) that does not
# vanish as physical distance shrinks to zero, which keeps the very short
# KJM-SBC suburban segment from becoming a mathematical singularity.
# K was hand-calibrated once so that the two metro termini (MAS-SBC) land
# at a plausible ~1500 daily rail passengers for this corridor.
GRAVITY_K = 0.0642
GRAVITY_DISTANCE_FLOOR_KM = 15.0


@dataclass
class Segment:
    index: int
    from_station: str
    to_station: str
    length_km: float


@dataclass
class Service:
    name: str
    stops: List[str]
    capacity: float
    cost_per_freq: float
    max_freq: int
    fare_per_km: float
    train_number: str = ""
    category: str = ""
    servable_od: List[Tuple[str, str]] = field(default_factory=list)
    route_segment_indices: List[int] = field(default_factory=list)


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_real_source(path: str = REAL_SOURCE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_segments(source: dict = None) -> List[Segment]:
    """Real consecutive-station segments with great-circle (haversine)
    distance computed from each station's real coordinates."""
    source = source or load_real_source()
    stations = source["stations"]
    segments = []
    for i in range(len(STATIONS) - 1):
        a, b = STATIONS[i], STATIONS[i + 1]
        sa, sb = stations[a], stations[b]
        dist = _haversine_km(sa["lon"], sa["lat"], sb["lon"], sb["lat"])
        segments.append(Segment(i, a, b, round(dist, 1)))
    return segments


def station_distance(a: str, b: str, segments: List[Segment]) -> float:
    ia, ib = STATIONS.index(a), STATIONS.index(b)
    assert ia < ib
    return sum(seg.length_km for seg in segments[ia:ib])


def build_od_demand(segments: List[Segment], seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthetic O-D demand via a gravity model calibrated with real
    (approximate, 2011 census) town/city populations -- see module
    docstring. Small multiplicative noise is added so the dataset is not
    perfectly deterministic in shape.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for o, d in combinations(STATIONS, 2):
        dist = station_distance(o, d, segments)
        eff_dist = max(dist, GRAVITY_DISTANCE_FLOOR_KM)
        pop_o, pop_d = STATION_POPULATION[o], STATION_POPULATION[d]
        base_demand = GRAVITY_K * math.sqrt(pop_o * pop_d) / eff_dist
        demand = max(10, round(base_demand * rng.uniform(0.85, 1.15)))
        rows.append({"origin": o, "destination": d, "distance_km": round(dist, 1),
                      "demand": int(demand)})
    return pd.DataFrame(rows)


def build_services(segments: List[Segment], source: dict = None) -> Dict[str, Service]:
    source = source or load_real_source()
    trains = source["trains"]
    corridor_stops = source["corridor_stops"]

    services: Dict[str, Service] = {}
    for train_number, meta in REAL_TRAINS.items():
        category = meta["category"]
        stops = corridor_stops[train_number]
        # keep stops in real corridor order regardless of source ordering
        stops = [s for s in STATIONS if s in stops]

        svc = Service(
            name=meta["display"],
            stops=stops,
            capacity=CAPACITY_BY_CATEGORY[category],
            cost_per_freq=round(COST_PER_KM_BY_CATEGORY[category] * trains[train_number]["distance"], 1),
            max_freq=MAX_FREQ_BY_CATEGORY[category],
            fare_per_km=FARE_PER_KM_BY_CATEGORY[category],
            train_number=train_number,
            category=category,
        )
        svc.servable_od = [(o, d) for o, d in combinations(stops, 2)]
        i0, i1 = STATIONS.index(stops[0]), STATIONS.index(stops[-1])
        svc.route_segment_indices = list(range(i0, i1))
        services[svc.name] = svc
    return services


def od_segment_incidence(o: str, d: str, segments: List[Segment]) -> List[int]:
    i0, i1 = STATIONS.index(o), STATIONS.index(d)
    return list(range(i0, i1))


def build_assignment_matrix(services: Dict[str, Service], segments: List[Segment]):
    """Builds the linear relation L = A x.

    x is the stacked vector of passenger-assignment decision variables
    y_{s,od} (flow of O-D pair ``od`` carried on service ``s``), ordered as
    ``columns`` below. L is the vector of segment loads, one entry per
    (service, segment) pair that the service actually traverses, ordered as
    ``rows`` below. A[row, col] = 1 iff variable ``columns[col]`` contributes
    load to ``rows[row]``.
    """
    columns: List[Tuple[str, str, str]] = []
    for svc in services.values():
        for (o, d) in svc.servable_od:
            columns.append((svc.name, o, d))

    rows: List[Tuple[str, int]] = []
    for svc in services.values():
        for seg_idx in svc.route_segment_indices:
            rows.append((svc.name, seg_idx))

    A = np.zeros((len(rows), len(columns)), dtype=int)
    row_index = {r: i for i, r in enumerate(rows)}
    for col_idx, (sname, o, d) in enumerate(columns):
        od_segs = set(od_segment_incidence(o, d, segments))
        for seg_idx in od_segs:
            key = (sname, seg_idx)
            if key in row_index:
                A[row_index[key], col_idx] = 1
    return A, rows, columns


def write_csvs(out_dir: str = DATA_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    source = load_real_source()
    segments = build_segments(source)
    services = build_services(segments, source)
    od_df = build_od_demand(segments)

    stations_df = pd.DataFrame([
        {"station": s, "name": source["stations"][s]["name"],
         "lat": source["stations"][s]["lat"], "lon": source["stations"][s]["lon"],
         "position_km": sum(seg.length_km for seg in segments[:STATIONS.index(s)])}
        for s in STATIONS
    ])
    segments_df = pd.DataFrame(
        [{"segment_index": s.index, "from": s.from_station, "to": s.to_station, "length_km": s.length_km}
         for s in segments]
    )
    services_rows = []
    for svc in services.values():
        services_rows.append({
            "service": svc.name,
            "train_number": svc.train_number,
            "category": svc.category,
            "stops": "-".join(svc.stops),
            "capacity_per_freq": svc.capacity,
            "cost_per_freq": svc.cost_per_freq,
            "fare_per_km": svc.fare_per_km,
            "max_freq": svc.max_freq,
        })
    services_df = pd.DataFrame(services_rows)

    stations_df.to_csv(os.path.join(out_dir, "stations.csv"), index=False)
    segments_df.to_csv(os.path.join(out_dir, "segments.csv"), index=False)
    od_df.to_csv(os.path.join(out_dir, "od_demand.csv"), index=False)
    services_df.to_csv(os.path.join(out_dir, "services.csv"), index=False)


def load_all():
    """Convenience loader used by the rest of the pipeline."""
    source = load_real_source()
    segments = build_segments(source)
    services = build_services(segments, source)
    od_df = build_od_demand(segments)
    return segments, services, od_df


if __name__ == "__main__":
    write_csvs()
    print(f"Real-data-derived network CSVs written to {DATA_DIR}")
