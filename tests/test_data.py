"""Sanity checks for every CSV in data/. Run: pytest tests/"""
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"

# file -> (numeric columns, text columns). Every listed column must have no nulls.
SPECS = {
    "penguins": (["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"], ["species", "island", "sex"]),
    "houston_weather": (["temp_high_f", "temp_low_f", "rain_in", "wind_mph"], ["date"]),
    "earthquakes": (["latitude", "longitude", "depth_km", "magnitude"], ["date", "place"]),
    "exoplanets": (["year_discovered", "orbit_days", "radius_earths", "mass_earths", "distance_parsecs"], ["planet", "discovery_method"]),
    "gapminder": (["year", "population", "life_expect", "fertility"], ["country"]),
    "us_states": (["year", "population", "births", "deaths", "net_migration"], ["state"]),
    "movies": (["year", "budget_2013_usd", "domestic_gross_2013_usd", "worldwide_gross_2013_usd"], ["title", "bechdel_result"]),
    "hurricanes": (["year", "month", "days_active", "max_wind_mph", "min_pressure_mb"], ["name", "made_landfall"]),
}


def test_every_csv_is_covered():
    assert {p.stem for p in DATA.glob("*.csv")} == set(SPECS)


@pytest.mark.parametrize("name", SPECS)
def test_csv(name):
    path = DATA / f"{name}.csv"
    numeric, text = SPECS[name]
    df = pd.read_csv(path)
    assert 100 <= len(df) <= 2000
    assert 4 <= df.shape[1] <= 10
    assert path.stat().st_size < 1_000_000
    assert df[numeric + text].notna().all().all()
    assert not df.isna().any().any()
    for c in numeric:
        assert pd.api.types.is_numeric_dtype(df[c]), c
    assert all(c == c.lower() and " " not in c for c in df.columns)
    path.read_bytes().decode("utf-8")
