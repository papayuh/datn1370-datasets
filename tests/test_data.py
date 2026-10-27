"""Sanity checks for every CSV in data/ (including data/synthetic/). Run: pytest tests/"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# scripts/synth.py is not a package, so load it by path
_spec = importlib.util.spec_from_file_location("synth", ROOT / "scripts" / "synth.py")
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)

# file (path under data/, no extension) -> (numeric columns, text columns). Every listed column must have no nulls.
SPECS = {
    "penguins": (["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"], ["species", "island", "sex"]),
    "houston_weather": (["temp_high_f", "temp_low_f", "rain_in", "wind_mph"], ["date"]),
    "earthquakes": (["latitude", "longitude", "depth_km", "magnitude"], ["date", "place"]),
    "exoplanets": (["year_discovered", "orbit_days", "radius_earths", "mass_earths", "distance_parsecs"], ["planet", "discovery_method"]),
    "gapminder": (["year", "population", "life_expect", "fertility"], ["country"]),
    "us_states": (["year", "population", "births", "deaths", "net_migration"], ["state"]),
    "movies": (["year", "budget_2013_usd", "domestic_gross_2013_usd", "worldwide_gross_2013_usd"], ["title", "bechdel_result"]),
    "hurricanes": (["year", "month", "days_active", "max_wind_mph", "min_pressure_mb"], ["name", "made_landfall"]),
    "synthetic/coffee_shop_sales": (["hot_drinks", "iced_drinks", "pastries", "revenue_usd", "temp_high_f"], ["date", "shop"]),
    "synthetic/gym_members": (["age", "join_month", "visits_month_1", "visits_month_6", "classes_per_month"], ["member_id", "plan", "still_active"]),
    "synthetic/streaming_songs": (["release_year", "duration_sec", "tempo_bpm", "energy", "streams_millions"], ["title", "artist", "genre"]),
    "synthetic/rideshare_trips": (["hour", "distance_miles", "duration_min", "fare_usd", "tip_usd", "rating"], ["date", "day_of_week", "pickup_zone"]),
    "synthetic/used_cars": (["year", "mileage", "price_usd", "owners"], ["make", "body_type", "fuel", "color"]),
    "synthetic/sneaker_resale": (["release_year", "retail_usd", "resale_usd", "pairs_released_thousands"], ["brand", "model", "collab", "colorway"]),
    "synthetic/student_budgets": (["work_hours_week", "monthly_income_usd", "rent_usd", "food_usd", "transport_usd", "entertainment_usd", "savings_usd"], ["student_id", "year_in_school", "living"]),
    "synthetic/social_media_posts": (["followers_thousands", "hour_posted", "caption_words", "hashtags", "likes", "comments", "shares"], ["creator", "post_type", "day_of_week"]),
    "synthetic/video_game_sales": (["release_year", "critic_score", "price_usd", "units_sold_millions"], ["title", "platform", "genre", "studio_size"]),
    "synthetic/fast_food_menu": (["calories", "fat_g", "carbs_g", "protein_g", "sodium_mg", "price_usd"], ["chain", "item", "category"]),
}


def test_every_csv_is_covered():
    assert {p.relative_to(DATA).with_suffix("").as_posix() for p in DATA.rglob("*.csv")} == set(SPECS)


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


@pytest.mark.parametrize("fn", synth.DATASETS, ids=lambda f: f.__name__)
def test_synthetic_files_match_script(fn):
    df = fn(np.random.default_rng([1370, synth.DATASETS.index(fn)]))
    assert df.to_csv(index=False) == (DATA / "synthetic" / f"{fn.__name__}.csv").read_text()
