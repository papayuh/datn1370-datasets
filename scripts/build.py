"""Download and clean every dataset in data/. Run: python scripts/build.py [name ...]"""
import html
import io
import sys
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data"


def get(url):
    r = requests.get(url, timeout=180, headers={"User-Agent": "datn1370-datasets"})
    r.raise_for_status()
    return r


def penguins():
    # CC0, Palmer Station LTER via seaborn-data
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
    df = df.dropna().reset_index(drop=True)
    return df.astype({"flipper_length_mm": int, "body_mass_g": int})


def houston_weather():
    # NOAA GHCN-Daily, Hobby Airport. Raw units: tenths of deg C, tenths of mm, tenths of m/s.
    raw = pd.read_csv(
        io.StringIO(get("https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00012918.csv").text),
        usecols=["DATE", "TMAX", "TMIN", "PRCP", "AWND"],
    )
    raw = raw[(raw.DATE >= "2022-01-01") & (raw.DATE <= "2024-12-31")].dropna()
    df = pd.DataFrame({
        "date": raw.DATE,
        "temp_high_f": (raw.TMAX / 10 * 9 / 5 + 32).round(1),
        "temp_low_f": (raw.TMIN / 10 * 9 / 5 + 32).round(1),
        "rain_in": (raw.PRCP / 10 / 25.4).round(2),
        "wind_mph": (raw.AWND / 10 * 2.23694).round(1),
    })
    month = pd.to_datetime(df.date).dt.month
    df["season"] = month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                              6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"})
    df["rain_status"] = df.rain_in.gt(0).map({True: "rain", False: "no measurable rain"})
    return df.reset_index(drop=True)


def earthquakes():
    # USGS ComCat, magnitude 5+, calendar year 2024
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2024-01-01"
           "&endtime=2025-01-01&minmagnitude=5&orderby=time-asc")
    raw = pd.read_csv(io.StringIO(get(url).text))
    raw = raw[raw.type == "earthquake"]
    df = pd.DataFrame({
        "date": raw.time.str[:10],
        "latitude": raw.latitude.round(3),
        "longitude": raw.longitude.round(3),
        "depth_km": raw.depth.round(1),
        "magnitude": raw.mag,
        "place": raw.place,
    })
    df = df.dropna().reset_index(drop=True)
    # Conventional depth classes; magnitude bands use the reported magnitude (unitless).
    df["depth_band"] = pd.cut(df.depth_km, [-float("inf"), 70, 300, float("inf")],
                              labels=["shallow", "intermediate", "deep"], right=False)
    df["magnitude_band"] = pd.cut(df.magnitude, [5, 6, 7, float("inf")],
                                  labels=["5.0-5.9", "6.0-6.9", "7.0+"], right=False)
    return df


def exoplanets():
    # NASA Exoplanet Archive, composite planet table
    url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query=select+pl_name,discoverymethod,"
           "disc_year,pl_orbper,pl_rade,pl_bmasse,sy_dist+from+pscomppars")
    raw = pd.read_csv(io.StringIO(get(url).text)).dropna()
    df = raw.rename(columns={
        "pl_name": "planet", "discoverymethod": "discovery_method", "disc_year": "year_discovered",
        "pl_orbper": "orbit_days", "pl_rade": "radius_earths", "pl_bmasse": "mass_earths",
        "sy_dist": "distance_parsecs",
    })
    for c in ["orbit_days", "radius_earths", "mass_earths", "distance_parsecs"]:
        df[c] = df[c].round(3)
    df["discovery_decade"] = ((df.year_discovered // 10) * 10).astype(str) + "s"
    # Rank methods after requiring complete measurements; rare sampled groups can remain small.
    df = df[df.discovery_method.isin(df.discovery_method.value_counts().head(4).index)]
    return df.sample(n=min(1500, len(df)), random_state=1).sort_values("planet").reset_index(drop=True)


def gapminder():
    # Gapminder Foundation (CC-BY 4.0) via vega-datasets
    raw = pd.read_json("https://raw.githubusercontent.com/vega/vega-datasets/main/data/gapminder.json")
    df = raw.rename(columns={"pop": "population"}).drop(columns="cluster").reset_index(drop=True)
    # Geographic continents (Central America/Caribbean in North America; Georgia/Turkey in Asia).
    groups = {
        "Africa": "Egypt|Kenya|Nigeria|Rwanda|South Africa",
        "Asia": "Afghanistan|Bangladesh|China|Georgia|Hong Kong, China|India|Indonesia|Iran|Iraq|Israel|Japan|Lebanon|North Korea|Pakistan|Philippines|Saudi Arabia|South Korea|Turkey",
        "Europe": "Austria|Belgium|Croatia|Finland|France|Germany|Greece|Iceland|Ireland|Italy|Netherlands|Norway|Poland|Portugal|Spain|Switzerland|United Kingdom",
        "North America": "Bahamas|Barbados|Canada|Costa Rica|Cuba|Dominican Republic|El Salvador|Grenada|Haiti|Jamaica|Mexico|United States",
        "South America": "Argentina|Bolivia|Brazil|Chile|Colombia|Ecuador|Peru|Venezuela",
        "Oceania": "Australia|New Zealand",
    }
    lookup = {country: continent for continent, countries in groups.items() for country in countries.split("|")}
    df["continent"] = df.country.map(lookup)
    if df.continent.isna().any():
        raise ValueError("Add new source countries to the continent lookup before rebuilding")
    return df


def us_states():
    # Census Population Estimates, state totals. 2010-2019 and 2020-2024 vintages, stacked.
    base = "https://www2.census.gov/programs-surveys/popest/datasets/"
    urls = [base + "2010-2019/national/totals/nst-est2019-alldata.csv",
            base + "2020-2024/state/totals/NST-EST2024-ALLDATA.csv"]
    frames = []
    for url in urls:
        raw = pd.read_csv(io.StringIO(get(url).text))
        raw = raw[raw.SUMLEV == 40]
        raw = raw[raw.NAME != "Puerto Rico"]  # keep 50 states + DC
        for year in range(2011, 2025):  # 2010 births/deaths cover Apr-Dec only
            cols = {f"POPESTIMATE{year}": "population", f"BIRTHS{year}": "births",
                    f"DEATHS{year}": "deaths", f"NETMIG{year}": "net_migration"}
            if not all(c in raw for c in cols):
                continue
            d = raw[["NAME", *cols]].rename(columns={"NAME": "state", **cols})
            d.insert(1, "year", year)
            frames.append(d)
    df = pd.concat(frames)
    # US Census Bureau's four regions, including DC in the South.
    regions = {
        "Northeast": "Connecticut|Maine|Massachusetts|New Hampshire|Rhode Island|Vermont|New Jersey|New York|Pennsylvania",
        "Midwest": "Indiana|Illinois|Michigan|Ohio|Wisconsin|Iowa|Kansas|Minnesota|Missouri|Nebraska|North Dakota|South Dakota",
        "South": "Delaware|District of Columbia|Florida|Georgia|Maryland|North Carolina|South Carolina|Virginia|West Virginia|Alabama|Kentucky|Mississippi|Tennessee|Arkansas|Louisiana|Oklahoma|Texas",
        "West": "Arizona|Colorado|Idaho|Montana|Nevada|New Mexico|Utah|Wyoming|Alaska|California|Hawaii|Oregon|Washington",
    }
    lookup = {state: region for region, states in regions.items() for state in states.split("|")}
    df["region"] = df.state.map(lookup)
    if df.region.isna().any():
        raise ValueError("Add new source states to the Census region lookup before rebuilding")
    df["migration_direction"] = df.net_migration.map(lambda n: "net inflow" if n > 0 else "net outflow" if n < 0 else "no change")
    return df.sort_values(["state", "year"]).reset_index(drop=True)


def movies():
    # FiveThirtyEight Bechdel test data (CC-BY 4.0); dollars are 2013 dollars
    raw = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv")
    df = pd.DataFrame({
        "title": raw.title.map(html.unescape),
        "year": raw.year,
        "bechdel_result": raw.binary.str.lower(),
        "budget_2013_usd": raw["budget_2013$"],
        "domestic_gross_2013_usd": raw["domgross_2013$"],
        "worldwide_gross_2013_usd": raw["intgross_2013$"],
    })
    df = df.dropna().astype({c: "int64" for c in df.columns if c.endswith("usd")}).reset_index(drop=True)
    df["decade"] = ((df.year // 10) * 10).astype(str) + "s"
    return df


def hurricanes():
    # NOAA HURDAT2 Atlantic best track: collapse each storm's 6-hourly rows into one row
    text = get("https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt").text
    storms, cur = [], None
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if f[0][:2] == "AL" and len(f[0]) == 8:
            cur = {"name": f[1].title(), "rows": []}
            storms.append(cur)
        else:
            cur["rows"].append(f)
    # HURDAT2 also names some depressions by number; these are not named storms.
    numbers = {"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
               "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
               "Eighteen", "Nineteen", "Twenty", "Twenty-One", "Twenty-Two", "Twenty-Three",
               "Twenty-Four", "Twenty-Five", "Twenty-Six", "Twenty-Seven", "Twenty-Eight",
               "Twenty-Nine", "Thirty", "Thirty-One"}
    out = []
    for s in storms:
        r = s["rows"]
        wind = [int(x[6]) for x in r]
        pres = [int(x[7]) for x in r if int(x[7]) > 0]
        if s["name"] in {"Unnamed", *numbers} or max(wind) < 0 or not pres:
            continue
        start, end = pd.to_datetime(r[0][0]), pd.to_datetime(r[-1][0])
        out.append({
            "name": s["name"], "year": start.year, "month": start.month,
            "days_active": (end - start).days + 1,
            "max_wind_mph": round(max(wind) * 1.15078), "min_pressure_mb": min(pres),
            "made_landfall": any(x[2] == "L" for x in r),
        })
    df = pd.DataFrame(out)
    return df[df.year >= 1980].reset_index(drop=True)


DATASETS = [penguins, houston_weather, earthquakes, exoplanets, gapminder, us_states, movies, hurricanes]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    want = sys.argv[1:]
    for fn in DATASETS:
        if want and fn.__name__ not in want:
            continue
        df = fn()
        df.to_csv(OUT / f"{fn.__name__}.csv", index=False, encoding="utf-8", lineterminator="\n")
        print(f"{fn.__name__}: {len(df)} rows x {df.shape[1]} cols")
