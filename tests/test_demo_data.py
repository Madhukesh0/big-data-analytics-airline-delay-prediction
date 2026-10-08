"""Demo generator tests: determinism, class balance, cancelled rows."""

from __future__ import annotations

from skypredict.data.demo_data import generate_demo_data


def test_deterministic_same_seed():
    f1, w1 = generate_demo_data(days=5, flights_per_day=20, seed=123)
    f2, w2 = generate_demo_data(days=5, flights_per_day=20, seed=123)
    pd = f1["flight_date"].dtype
    assert f1.equals(f2)
    assert w1.equals(w2)


def test_has_both_classes_and_outcomes():
    flights, _w = generate_demo_data(days=10, flights_per_day=40, seed=42)
    delayed = flights["arrival_delay_minutes"] >= 15
    on_time = flights["arrival_delay_minutes"].notna() & ~delayed
    assert delayed.any(), "generator should produce delayed flights"
    assert on_time.any(), "generator should produce on-time flights"
    assert (flights["cancelled"] == 1).any() or (flights["diverted"] == 1).any()
    # cancelled/diverted flights carry no target
    assert flights.loc[flights["cancelled"] == 1, "arrival_delay_minutes"].isna().all()
