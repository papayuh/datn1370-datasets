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
    return df.dropna().reset_index(drop=True)


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
    # keep the 4 methods with plenty of planets so charts by method are readable
    df = df[df.discovery_method.isin(df.discovery_method.value_counts().head(4).index)]
    return df.sample(n=min(1500, len(df)), random_state=1).sort_values("planet").reset_index(drop=True)


def gapminder():
    # Gapminder Foundation (CC-BY 4.0) via vega-datasets
    raw = pd.read_json("https://raw.githubusercontent.com/vega/vega-datasets/main/data/gapminder.json")
    return raw.rename(columns={"pop": "population"}).drop(columns="cluster").reset_index(drop=True)


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
    return df.dropna().astype({c: "int64" for c in df.columns if c.endswith("usd")}).reset_index(drop=True)


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
    out = []
    for s in storms:
        r = s["rows"]
        wind = [int(x[6]) for x in r]
        pres = [int(x[7]) for x in r if int(x[7]) > 0]
        if s["name"] == "Unnamed" or max(wind) < 0 or not pres:
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
        df.to_csv(OUT / f"{fn.__name__}.csv", index=False)
        print(f"{fn.__name__}: {len(df)} rows x {df.shape[1]} cols")
