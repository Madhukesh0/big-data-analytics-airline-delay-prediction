"""Deterministic synthetic demo-data generator.

Produces a synthetic flight dataset (and optional matching synthetic weather
observations) so the project runs with no internet access and no external
files. The generator is seeded, so the same parameters always produce the
same rows.

IMPORTANT: this data is SYNTHETIC DEMO DATA. It must always be labelled as
such in the UI, output files, and reports. It is not real airline data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DELAY_THRESHOLD_MINUTES, RANDOM_SEED

SYNTHETIC_SOURCE_LABEL = "synthetic_demo"

CARRIER_DELAY_PROPENSITY = {  # base logit contribution per carrier
    "AA": 0.35,
    "DL": -0.55,
    "UA": 0.45,
    "WN": -0.30,
    "AS": -0.70,
}
AIRPORT_DELAY_PROPENSITY = {
    "ATL": -0.15,
    "DFW": 0.25,
    "DEN": 0.10,
    "ORD": 0.60,
    "LAX": -0.05,
    "SEA": -0.35,
    "SFO": 0.75,
    "JFK": 0.45,
    "MIA": 0.15,
    "BOS": 0.05,
}
AIRPORTS = list(AIRPORT_DELAY_PROPENSITY)
CARRIERS = list(CARRIER_DELAY_PROPENSITY)


def generate_demo_data(
    days: int = 120,
    flights_per_day: int = 60,
    seed: int = RANDOM_SEED,
    start_date: str = "2026-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic flights plus synthetic weather observations.

    Returns ``(flights, weather)`` in canonical format. Weather affects the
    simulated delay probability (rainy/windy days delay more), so the weather
    features genuinely carry signal in the demo data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=days, freq="D")

    weather = _generate_weather(dates, rng)

    rows = []
    for day in dates:
        month = day.month
        for _ in range(flights_per_day):
            carrier = CARRIERS[int(rng.integers(0, len(CARRIERS)))]
            origin, destination = _pick_route(rng)
            hour = int(rng.integers(5, 22))
            minute = int(rng.integers(0, 60))
            dep_minute = hour * 60 + minute
            distance = float(rng.integers(250, 2600))
            duration = _estimate_duration(distance, dep_minute, rng)

            w = weather[
                (weather["airport"] == origin)
                & (weather["observation_date"] == day)
            ].iloc[0]

            p_delay = _delay_probability(
                carrier=carrier,
                origin=origin,
                destination=destination,
                dep_minute=dep_minute,
                month=month,
                dow=day.dayofweek,
                precip_mm=float(w["precipitation_mm"]),
                wind_kmh=float(w["wind_speed_kmh"]),
            )
            is_cancelled = int(rng.random() < 0.012)
            is_diverted = int(not is_cancelled and rng.random() < 0.003)
            is_delay = 0
            delay_minutes = float(rng.uniform(-15, 14))
            if not is_cancelled and not is_diverted and rng.random() < p_delay:
                is_delay = 1
                delay_minutes = float(DELAY_THRESHOLD_MINUTES + rng.gamma(1.8, 32.0))

            rows.append(
                {
                    "flight_date": day,
                    "carrier": carrier,
                    "origin": origin,
                    "destination": destination,
                    "sched_dep_minute": dep_minute,
                    "sched_arr_minute": (dep_minute + int(duration)) % 1440,
                    "sched_duration_minutes": duration,
                    "arrival_delay_minutes": None if (is_cancelled or is_diverted) else round(delay_minutes, 1),
                    "cancelled": is_cancelled,
                    "diverted": is_diverted,
                    "distance": distance,
                    "flight_number": f"{carrier}{int(rng.integers(100, 3000))}",
                }
            )

    flights = pd.DataFrame(rows)
    flights["arrival_delay_minutes"] = flights["arrival_delay_minutes"].astype("float64")
    flights["is_delayed"] = pd.Series(pd.NA, index=flights.index, dtype="Int64")
    has_outcome = (
        flights["arrival_delay_minutes"].notna()
        & (flights["cancelled"] == 0)
        & (flights["diverted"] == 0)
    )
    flights.loc[has_outcome, "is_delayed"] = (
        flights.loc[has_outcome, "arrival_delay_minutes"] >= DELAY_THRESHOLD_MINUTES
    ).astype("int64")
    return flights, weather


def _generate_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """One synthetic weather observation per airport per day at 00:30 local.

    The observation timestamp is deliberately before any flight's
    prediction cutoff (departure minus 2 hours, flights start at 05:00),
    which keeps the weather join point-in-time safe.
    """
    rows = []
    base_temp = rng.uniform(2, 24)
    for i, day in enumerate(dates):
        seasonal = 12 * np.sin(2 * np.pi * (day.dayofyear - 80) / 365)
        for airport in AIRPORTS:
            temp = base_temp + seasonal + rng.normal(0, 4)
            wind = float(max(0.0, rng.gamma(2.0, 8.0)))
            rainy = rng.random() < 0.22
            precip = float(rng.exponential(2.5)) if rainy else 0.0
            visibility = float(np.clip(rng.normal(16, 4) - (6 if rainy else 0), 1, 25))
            rows.append(
                {
                    "airport": airport,
                    "observation_date": day,
                    "observation_timestamp": day + pd.Timedelta(minutes=30),
                    "temperature_c": round(float(temp), 1),
                    "wind_speed_kmh": round(wind, 1),
                    "precipitation_mm": round(precip, 1),
                    "visibility_km": round(visibility, 1),
                }
            )
    out = pd.DataFrame(rows)
    out["observation_timestamp"] = out["observation_timestamp"].astype(str)
    return out


def _pick_route(rng: np.random.Generator) -> tuple[str, str]:
    origin = AIRPORTS[int(rng.integers(0, len(AIRPORTS)))]
    destination = AIRPORTS[int(rng.integers(0, len(AIRPORTS)))]
    while destination == origin:
        destination = AIRPORTS[int(rng.integers(0, len(AIRPORTS)))]
    return origin, destination


def _estimate_duration(distance_km_like: float, dep_minute: int, rng: np.random.Generator) -> float:
    """Rough scheduled duration: distance/8 min plus taxi, capped at 6h."""
    duration = distance_km_like / 8.0 + 30 + rng.normal(0, 5)
    return float(np.clip(round(duration), 45, 360))


def _delay_probability(
    carrier: str,
    origin: str,
    destination: str,
    dep_minute: int,
    month: int,
    dow: int,
    precip_mm: float,
    wind_kmh: float,
) -> float:
    """Delay probability from a small additive logit model (synthetic).

    The signal is deliberately strong enough to be learnable but not trivial:
    time-of-day buildup, carrier/hub effects, day-of-week and weather all
    contribute, so the generator has a Bayes-optimal ROC-AUC around 0.80.
    """
    hour = dep_minute / 60.0
    logit = -5.0
    logit += CARRIER_DELAY_PROPENSITY[carrier]
    logit += 0.6 * AIRPORT_DELAY_PROPENSITY[origin]
    logit += 0.4 * AIRPORT_DELAY_PROPENSITY[destination]
    # Delays build through the operating day (strong, smooth, monotone signal).
    logit += 0.42 * max(hour - 5.5, 0.0)
    logit += 0.28 if dow == 4 else 0.0        # Friday (Mon=0)
    logit += 0.22 if dow == 6 else 0.0        # Sunday
    logit += 0.30 if month in (6, 7, 12) else 0.0
    # Weather: precipitation dominates, wind adds a smaller effect.
    logit += 1.35 * min(precip_mm / 4.0, 1.0)
    logit += 0.05 * min(wind_kmh, 45.0)
    return float(1.0 / (1.0 + np.exp(-logit)))
