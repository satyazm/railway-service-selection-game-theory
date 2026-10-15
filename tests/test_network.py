import numpy as np

from src.network import (
    REAL_TRAINS,
    STATION_POPULATION,
    STATIONS,
    build_assignment_matrix,
    build_od_demand,
    build_segments,
    build_services,
    load_real_source,
    od_segment_incidence,
    station_distance,
)


def test_real_source_covers_all_stations_and_trains():
    source = load_real_source()
    for s in STATIONS:
        assert s in source["stations"]
        assert s in STATION_POPULATION
    for train_number in REAL_TRAINS:
        assert train_number in source["trains"]
        assert train_number in source["corridor_stops"]


def test_segment_distances_are_real_and_plausible():
    # real haversine distance between Chennai Central and Bangalore City
    # (via the real corridor) should be in the right physical ballpark
    # (actual rail distance is ~362 km; great-circle is necessarily a bit
    # shorter since the line is not perfectly straight).
    segments = build_segments()
    total = sum(s.length_km for s in segments)
    assert 280 <= total <= 362


def test_segments_span_all_stations():
    segments = build_segments()
    assert len(segments) == len(STATIONS) - 1
    assert segments[0].from_station == STATIONS[0]
    assert segments[-1].to_station == STATIONS[-1]


def test_station_distance_matches_segment_sum():
    segments = build_segments()
    d = station_distance("MAS", "JTJ", segments)
    assert d == sum(s.length_km for s in segments[0:3])


def test_od_demand_positive_and_covers_all_pairs():
    segments = build_segments()
    od_df = build_od_demand(segments)
    n = len(STATIONS)
    assert len(od_df) == n * (n - 1) // 2
    assert (od_df["demand"] > 0).all()
    assert (od_df["distance_km"] > 0).all()


def test_services_servable_od_within_route():
    segments = build_segments()
    services = build_services(segments)
    for svc in services.values():
        for (o, d) in svc.servable_od:
            assert o in svc.stops and d in svc.stops
            assert svc.stops.index(o) < svc.stops.index(d)


def test_od_segment_incidence_matches_station_indices():
    segments = build_segments()
    segs = od_segment_incidence("KPD", "KJM", segments)
    assert segs == [2, 3, 4]


def test_assignment_matrix_shape_and_binary():
    segments = build_segments()
    services = build_services(segments)
    A, rows, columns = build_assignment_matrix(services, segments)
    assert A.shape == (len(rows), len(columns))
    assert set(np.unique(A)).issubset({0, 1})
    # every row (service,segment) must have at least one OD column that can load it
    assert (A.sum(axis=1) > 0).all()


def test_assignment_matrix_L_equals_Ax():
    """Directly checks the L = A x relation this module is built around."""
    segments = build_segments()
    services = build_services(segments)
    A, rows, columns = build_assignment_matrix(services, segments)

    rng = np.random.default_rng(0)
    x = rng.integers(0, 10, size=len(columns))
    L = A @ x

    # recompute L by brute force and compare
    for row_idx, (s, seg_idx) in enumerate(rows):
        expected = 0
        for col_idx, (sname, o, d) in enumerate(columns):
            if sname != s:
                continue
            od_segs = set(od_segment_incidence(o, d, segments))
            if seg_idx in od_segs:
                expected += x[col_idx]
        assert L[row_idx] == expected
