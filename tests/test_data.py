"""Sanity checks for every CSV in data/ (including data/synthetic/). Run: pytest tests/"""
import csv
import importlib.util
import io
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# scripts/synth.py is not a package, so load it by path
_spec = importlib.util.spec_from_file_location("synth", ROOT / "scripts" / "synth.py")
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)

_spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

# file (path under data/, no extension) -> (numeric columns, text columns). Every listed column must have no nulls.
SPECS = {
    "penguins": (["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"], ["species", "island", "sex"]),
    "houston_weather": (["temp_high_f", "temp_low_f", "rain_in", "wind_mph"], ["date", "season", "rain_status"]),
    "earthquakes": (["latitude", "longitude", "depth_km", "magnitude"], ["date", "place", "depth_band", "magnitude_band"]),
    "exoplanets": (["year_discovered", "orbit_days", "radius_earths", "mass_earths", "distance_parsecs"], ["planet", "discovery_method", "discovery_decade"]),
    "gapminder": (["year", "population", "life_expect", "fertility"], ["country", "continent"]),
    "us_states": (["year", "population", "births", "deaths", "net_migration"], ["state", "region", "migration_direction"]),
    "movies": (["year", "budget_2013_usd", "domestic_gross_2013_usd", "worldwide_gross_2013_usd"], ["title", "bechdel_result", "decade"]),
    "hurricanes": (["year", "month", "days_active", "max_wind_mph", "min_pressure_mb"], ["name", "made_landfall"]),
    "synthetic/coffee_shop_sales": (["hot_drinks", "iced_drinks", "pastries", "revenue_usd", "temp_high_f"], ["date", "shop", "day_of_week"]),
    "synthetic/gym_members": (["age", "join_month", "visits_month_1", "visits_month_6", "classes_per_month"], ["member_id", "plan", "still_active"]),
    "synthetic/streaming_songs": (["release_year", "duration_sec", "tempo_bpm", "energy", "streams_millions"], ["title", "artist", "genre", "release_decade"]),
    "synthetic/rideshare_trips": (["hour", "distance_miles", "duration_min", "fare_usd", "tip_usd", "rating"], ["date", "day_of_week", "pickup_zone"]),
    "synthetic/used_cars": (["year", "mileage", "price_usd", "owners"], ["make", "body_type", "fuel", "color"]),
    "synthetic/sneaker_resale": (["release_year", "retail_usd", "resale_usd", "pairs_released_thousands"], ["brand", "model", "collab", "colorway"]),
    "synthetic/student_budgets": (["work_hours_week", "monthly_income_usd", "rent_usd", "food_usd", "transport_usd", "entertainment_usd", "savings_usd"], ["student_id", "year_in_school", "living"]),
    "synthetic/social_media_posts": (["followers_thousands", "hour_posted", "caption_words", "hashtags", "likes", "comments", "shares"], ["creator", "post_type", "day_of_week"]),
    "synthetic/video_game_sales": (["release_year", "critic_score", "price_usd", "units_sold_millions"], ["title", "platform", "genre", "studio_size"]),
    "synthetic/fast_food_menu": (["calories", "fat_g", "carbs_g", "protein_g", "sodium_mg", "price_usd"], ["chain", "item", "category"]),
}


# Primary category (3-12 groups), second category (2-12), two measured quantities.
# Month/year are legitimate categorical axes, not measurements for the scatterplot.
CHARTS = {
    "penguins": ("species", "sex", "flipper_length_mm", "body_mass_g"),
    "houston_weather": ("season", "rain_status", "temp_high_f", "rain_in"),
    "earthquakes": ("depth_band", "magnitude_band", "depth_km", "magnitude"),
    "exoplanets": ("discovery_method", "discovery_decade", "radius_earths", "mass_earths"),
    "gapminder": ("continent", "year", "life_expect", "fertility"),
    "us_states": ("region", "migration_direction", "births", "deaths"),
    "movies": ("decade", "bechdel_result", "budget_2013_usd", "worldwide_gross_2013_usd"),
    "hurricanes": ("month", "made_landfall", "max_wind_mph", "min_pressure_mb"),
    "synthetic/coffee_shop_sales": ("shop", "day_of_week", "temp_high_f", "revenue_usd"),
    "synthetic/gym_members": ("plan", "still_active", "visits_month_1", "visits_month_6"),
    "synthetic/streaming_songs": ("genre", "release_decade", "duration_sec", "streams_millions"),
    "synthetic/rideshare_trips": ("pickup_zone", "day_of_week", "distance_miles", "fare_usd"),
    "synthetic/used_cars": ("body_type", "fuel", "mileage", "price_usd"),
    "synthetic/sneaker_resale": ("brand", "collab", "retail_usd", "resale_usd"),
    "synthetic/student_budgets": ("living", "year_in_school", "monthly_income_usd", "savings_usd"),
    "synthetic/social_media_posts": ("post_type", "day_of_week", "followers_thousands", "likes"),
    "synthetic/video_game_sales": ("genre", "studio_size", "critic_score", "units_sold_millions"),
    "synthetic/fast_food_menu": ("category", "chain", "calories", "sodium_mg"),
}

# Inclusive, deliberately broad domain limits, in the same order as SPECS' numeric columns.
# Counts need no suffix; legacy age/mileage/life_expect/fertility units are in README.
BOUNDS = {
    "penguins": [(25, 70), (10, 25), (150, 250), (2000, 7000)],
    "houston_weather": [(-10, 120), (-10, 100), (0, 20), (0, 100)],
    "earthquakes": [(-90, 90), (-180, 180), (0, 800), (5, 10)],
    "exoplanets": [(1980, 2030), (0.001, 1e7), (0.01, 100), (0.001, 1e5), (0.01, 1e5)],
    "gapminder": [(1955, 2005), (1, 2e9), (20, 100), (0, 12)],
    "us_states": [(2011, 2024), (1e5, 5e7), (1, 1e6), (1, 1e6), (-1e6, 1e6)],
    "movies": [(1970, 2013), (1, 1e9), (0, 5e9), (0, 1e10)],
    "hurricanes": [(1980, 2025), (1, 12), (1, 60), (39, 250), (800, 1050)],
    "synthetic/coffee_shop_sales": [(0, 1000), (0, 1000), (0, 1000), (0, 10000), (35, 103)],
    "synthetic/gym_members": [(18, 75), (1, 12), (1, 62), (0, 62), (0, 31)],
    "synthetic/streaming_songs": [(2000, 2025), (90, 420), (60, 190), (0, 1), (0.1, 10000)],
    "synthetic/rideshare_trips": [(0, 23), (0.5, 40), (3, 240), (5, 300), (0, 120), (1, 5)],
    "synthetic/used_cars": [(1965, 2025), (500, 300000), (1500, 100000), (1, 5)],
    "synthetic/sneaker_resale": [(2015, 2025), (50, 500), (40, 20000), (2, 800)],
    "synthetic/student_budgets": [(0, 40), (200, 5000), (0, 2000), (60, 1000), (0, 1000), (10, 1000), (-5000, 5000)],
    "synthetic/social_media_posts": [(0.5, 5000), (0, 23), (1, 200), (0, 15), (1, 1e7), (0, 1e6), (0, 1e6)],
    "synthetic/video_game_sales": [(2005, 2025), (0, 100), (0, 100), (0.01, 100)],
    "synthetic/fast_food_menu": [(40, 1800), (0, 200), (0, 450), (0, 450), (0, 10000), (0, 50)],
}


def test_every_csv_is_covered():
    assert {p.relative_to(DATA).with_suffix("").as_posix() for p in DATA.rglob("*.csv")} == set(SPECS)
    assert set(CHARTS) == set(BOUNDS) == set(SPECS)
    assert len(SPECS) == 18


def test_readme_menu_and_chart_picker_match_files():
    readme = (ROOT / "README.md").read_text()
    prefix = "https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/"
    assert set(re.findall(re.escape(prefix) + r'([^"\s]+)\.csv', readme)) == set(SPECS)
    for name in SPECS:
        df = pd.read_csv(DATA / f"{name}.csv")
        short_name = name.split("/")[-1]
        menu = next(line for line in readme.splitlines() if line.startswith(f"| **{short_name}** |"))
        cells = [cell.strip() for cell in menu.split("|")]
        assert cells[3] == f"{len(df):,} x {len(df.columns)}"
        assert re.findall(r"`([^`]+)`", cells[4]) == df.columns.tolist()
        picker = next(line for line in readme.splitlines() if line.startswith(f"| {short_name} |"))
        cells = [cell.strip() for cell in picker.split("|")]
        assert cells[2] == f"{len(df):,}"
        assert set(re.findall(r"`([^`]+)`", cells[3])) <= set(df.columns)
        assert len(cells[4].split("; ")) == 2


@pytest.mark.parametrize("name", SPECS)
def test_csv(name):
    path = DATA / f"{name}.csv"
    numeric, text = SPECS[name]
    df = pd.read_csv(path)
    assert 100 <= len(df) <= 2000
    assert set(df.columns) == set(numeric + text)
    assert 4 <= df.shape[1] <= 10
    assert path.stat().st_size < 1_000_000  # stricter than the 5 MB teaching limit
    assert df[numeric + text].notna().all().all()
    assert not df.isna().any().any()
    for c in numeric:
        assert pd.api.types.is_numeric_dtype(df[c]), c
    assert all(re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", c) for c in df.columns)
    content = path.read_bytes()
    assert b"\r" not in content, "CSV must use LF line endings"
    decoded = content.decode("utf-8")
    assert decoded.endswith("\n")
    assert all(len(row) == len(df.columns) for row in csv.reader(io.StringIO(decoded)))
    for c in text:
        if not pd.api.types.is_bool_dtype(df[c]):
            assert df[c].map(lambda value: isinstance(value, str) and value == value.strip() and bool(value)).all(), c
    if "date" in df:
        assert df.date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
        dates = pd.to_datetime(df.date, format="%Y-%m-%d", errors="raise")
        assert dates.notna().all()


@pytest.mark.parametrize("name", SPECS)
def test_chart_readiness(name):
    df = pd.read_csv(DATA / f"{name}.csv")
    category, second_category, x, y = CHARTS[name]
    assert 3 <= df[category].nunique() <= 12
    assert 2 <= df[second_category].nunique() <= 12
    # These two tables feed a bar/box plot and a stacked bar with only light aggregation.
    assert len(df.groupby(category)[y].mean()) >= 3
    assert pd.crosstab(df[category], df[second_category]).to_numpy().sum() == len(df)
    for c in (x, y):
        assert c in SPECS[name][0]
        assert df[c].nunique() >= 10, f"{c} must be a varying measurement, not an ID/category"
    for c, (low, high) in zip(SPECS[name][0], BOUNDS[name], strict=True):
        assert np.isfinite(df[c]).all(), c
        assert df[c].between(low, high).all(), f"{name}.{c} outside [{low}, {high}]"


def test_derived_categories_and_relationships():
    weather = pd.read_csv(DATA / "houston_weather.csv")
    months = pd.to_datetime(weather.date).dt.month
    expected = months.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"})
    assert weather.season.equals(expected.rename("season"))
    assert weather.rain_status.eq("rain").equals(weather.rain_in.gt(0))
    assert weather.temp_high_f.ge(weather.temp_low_f).all()
    quakes = pd.read_csv(DATA / "earthquakes.csv")
    assert quakes.depth_band.eq("shallow").equals(quakes.depth_km.lt(70))
    assert quakes.depth_band.eq("intermediate").equals(quakes.depth_km.ge(70) & quakes.depth_km.lt(300))
    assert quakes.depth_band.eq("deep").equals(quakes.depth_km.ge(300))
    expected = quakes.magnitude.map(lambda m: "5.0-5.9" if m < 6 else "6.0-6.9" if m < 7 else "7.0+")
    assert quakes.magnitude_band.equals(expected.rename("magnitude_band"))
    for name, year, decade in [("movies", "year", "decade"), ("exoplanets", "year_discovered", "discovery_decade"),
                               ("synthetic/streaming_songs", "release_year", "release_decade")]:
        df = pd.read_csv(DATA / f"{name}.csv")
        assert df[decade].equals((((df[year] // 10) * 10).astype(str) + "s").rename(decade))
    states = pd.read_csv(DATA / "us_states.csv")
    assert states.migration_direction.eq("net inflow").equals(states.net_migration.gt(0))
    assert states.migration_direction.eq("net outflow").equals(states.net_migration.lt(0))
    assert states.groupby("state").region.nunique().eq(1).all()
    assert states.loc[states.state.eq("District of Columbia"), "region"].eq("South").all()
    gap = pd.read_csv(DATA / "gapminder.csv")
    assert gap.groupby("country").continent.nunique().eq(1).all()
    coffee = pd.read_csv(DATA / "synthetic/coffee_shop_sales.csv")
    assert coffee.day_of_week.tolist() == [synth.DAYS[d] for d in pd.to_datetime(coffee.date).dt.dayofweek]
    budget = pd.read_csv(DATA / "synthetic/student_budgets.csv")
    expenses = budget[["rent_usd", "food_usd", "transport_usd", "entertainment_usd"]].sum(axis=1)
    assert budget.savings_usd.equals((budget.monthly_income_usd - expenses).rename("savings_usd"))
    assert budget.savings_usd.lt(0).any()  # documented shortfalls are not missing data


def test_earthquake_builder_band_boundaries(monkeypatch):
    raw = pd.DataFrame({
        "time": ["2024-01-01T00:00:00Z"] * 5, "latitude": 0, "longitude": 0,
        "depth": [69.9, 70, 299.9, 300, 640], "mag": [5, 5.9, 6, 6.9, 7],
        "place": "Test location", "type": "earthquake",
    })
    monkeypatch.setattr(build, "get", lambda url: SimpleNamespace(text=raw.to_csv(index=False)))
    df = build.earthquakes()
    assert df.depth_band.tolist() == ["shallow", "intermediate", "intermediate", "deep", "deep"]
    assert df.magnitude_band.tolist() == ["5.0-5.9", "5.0-5.9", "6.0-6.9", "6.0-6.9", "7.0+"]


def test_hurricanes_skip_numbered_and_unnamed_storms(monkeypatch):
    def storm(name, year=2024, wind=40):
        return f"AL012024, {name}, 1,\n{year}0801, 0000, L, TS, 20.0N, 50.0W, {wind}, 1000,\n"

    excluded = ["UNNAMED", "ONE", "TEN", "FIFTEEN", "TWENTY-ONE", "TWENTY-TWO", "THIRTY-ONE"]
    text = "".join(storm(name) for name in excluded)
    text += storm("ALBERTO", 1980) + storm("ALPHA", 2005) + storm("BETA", 2020) + storm("OLD", 1979)
    monkeypatch.setattr(build, "get", lambda url: SimpleNamespace(text=text))
    df = build.hurricanes()
    assert df.name.tolist() == ["Alberto", "Alpha", "Beta"]
    assert df.made_landfall.all()
    assert df.days_active.eq(1).all()
    actual = pd.read_csv(DATA / "hurricanes.csv")
    assert not actual.name.str.fullmatch(r"Unnamed|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty(?:-.*)?|Thirty(?:-.*)?").any()


@pytest.mark.parametrize("fn", synth.DATASETS, ids=lambda f: f.__name__)
def test_synthetic_files_match_script(fn):
    df = fn(np.random.default_rng([1370, synth.DATASETS.index(fn)]))
    assert df.to_csv(index=False, lineterminator="\n").encode("utf-8") == (DATA / "synthetic" / f"{fn.__name__}.csv").read_bytes()
