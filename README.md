# DATN 1370 datasets

Pick **one** dataset for Module 5. Ask your own question, make charts, save your Colab notebook to GitHub, submit the link.

All files are clean: no missing values, numbers are plain numbers, one row per thing, column names in snake_case. Load one with a single line in Colab:

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/penguins.csv")
df.head()
```

Then a chart in two more lines:

```python
import seaborn as sns
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g", hue="species")
```

## The menu

| dataset | one row is | rows x cols | columns | starter question | source + license | load it |
|---|---|---|---|---|---|---|
| **penguins** | One penguin | 333 x 7 | `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex` | Do heavier penguins always have longer flippers, or does it depend on species? | [Palmer Station LTER](https://github.com/allisonhorst/palmerpenguins) via seaborn-data. CC0 | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/penguins.csv")` |
| **houston_weather** | One day at Hobby Airport, 2022-2024 | 1,092 x 5 | `date`, `temp_high_f`, `temp_low_f`, `rain_in`, `wind_mph` | How much hotter are Houston summers getting, and when does it actually rain? | [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily), station USW00012918. US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/houston_weather.csv")` |
| **earthquakes** | One magnitude 5+ earthquake in 2024 | 1,507 x 6 | `date`, `latitude`, `longitude`, `depth_km`, `magnitude`, `place` | Are deeper earthquakes bigger or smaller? Where do the biggest ones cluster? | [USGS](https://earthquake.usgs.gov/fdsnws/event/1/). US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/earthquakes.csv")` |
| **exoplanets** | One confirmed planet outside our solar system (random 1,500 from the 4 most common discovery methods) | 1,500 x 7 | `planet`, `discovery_method`, `year_discovered`, `orbit_days`, `radius_earths`, `mass_earths`, `distance_parsecs` | Which discovery method finds which kind of planet? How did that change by year? | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/). US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/exoplanets.csv")` |
| **gapminder** | One country in one year (every 5 years, 1955-2005) | 682 x 5 | `year`, `country`, `population`, `life_expect`, `fertility` | Did every country get healthier at the same pace? Which ones broke the pattern? | [Gapminder Foundation](https://www.gapminder.org/data/) via vega-datasets. CC-BY 4.0, credit Gapminder | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/gapminder.csv")` |
| **us_states** | One state (or DC) in one year, 2011-2024 | 714 x 6 | `state`, `year`, `population`, `births`, `deaths`, `net_migration` | Which states are growing fastest, and is it births or people moving in? | [US Census Bureau Population Estimates](https://www.census.gov/programs-surveys/popest.html). US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/us_states.csv")` |
| **movies** | One movie, 1970-2013 | 1,776 x 6 | `title`, `year`, `bechdel_result`, `budget_2013_usd`, `domestic_gross_2013_usd`, `worldwide_gross_2013_usd` | Do movies that pass the Bechdel test make less money? Does budget predict gross? | [FiveThirtyEight](https://github.com/fivethirtyeight/data/tree/master/bechdel) (data from bechdeltest.com). CC-BY 4.0, credit FiveThirtyEight | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/movies.csv")` |
| **hurricanes** | One named Atlantic storm, 1980-2025 | 634 x 7 | `name`, `year`, `month`, `days_active`, `max_wind_mph`, `min_pressure_mb`, `made_landfall` | Are storms getting stronger over time? Which months are the worst? | [NOAA National Hurricane Center HURDAT2](https://www.nhc.noaa.gov/data/#hurdat). US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/hurricanes.csv")` |

## Notes

- `houston_weather`: days with a missing reading were dropped (1,092 of 1,096 days kept). Temperatures converted to F, rain to inches, wind to mph.
- `exoplanets`: only planets with every measurement filled in. Sampled down to 1,500 rows to keep it manageable.
- `gapminder`: the `cluster` column from the source was dropped.
- `us_states`: 50 states + DC, Puerto Rico left out. 2010 left out because its births/deaths only cover Apr-Dec. Vintage 2010-2019 and 2020-2024 estimates are stacked, so there can be a small jump at 2020.
- `movies`: dollar columns are in 2013 dollars. Movies missing budget or gross were dropped.
- `hurricanes`: unnamed storms and storms before 1980 dropped. `max_wind_mph` and `min_pressure_mb` are the peak over the storm's life.

## Where it came from

Every file is rebuilt from its source by `python scripts/build.py` (needs `pandas` and `requests`). Data retrieved 2026-10-27. Run `pytest tests/` to check the files.

Swaps from the original wish list, because there is no clearly open source for the original topic: NBA player stats, Spotify top songs, video game sales and fast food nutrition are all scraped or Kaggle-only with unclear licenses, so they became exoplanets, earthquakes, hurricanes and gapminder. US states income needs a Census API key, so `us_states` is population, births, deaths and migration instead.
