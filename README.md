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

## Synthetic (made-up) datasets

These are generated by `scripts/synth.py`. Every number and name in them is invented. Nothing here describes a real business, person, product or place, so never cite them as facts. They are here because the real menu lacks everyday consumer topics. Same rules as above: clean, no missing values, snake_case columns.

| dataset | one row is | rows x cols | columns | starter question | source + license | load it |
|---|---|---|---|---|---|---|
| **coffee_shop_sales** | One day at one shop of a small coffee chain, 2025 | 1,095 x 7 | `date`, `shop`, `hot_drinks`, `iced_drinks`, `pastries`, `revenue_usd`, `temp_high_f` | Which shop does best on weekends, and does hot weather change what people order? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/coffee_shop_sales.csv")` |
| **gym_members** | One member of a gym who joined in 2025 | 1,200 x 8 | `member_id`, `age`, `plan`, `join_month`, `visits_month_1`, `visits_month_6`, `classes_per_month`, `still_active` | Do people who join in January stick around as long as everyone else? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/gym_members.csv")` |
| **streaming_songs** | One song on a streaming service | 1,000 x 8 | `title`, `artist`, `genre`, `release_year`, `duration_sec`, `tempo_bpm`, `energy`, `streams_millions` | Have songs gotten shorter over the years, and does that matter for how many streams they get? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/streaming_songs.csv")` |
| **rideshare_trips** | One trip with a rideshare app in one city, 2025 | 1,500 x 9 | `date`, `hour`, `day_of_week`, `pickup_zone`, `distance_miles`, `duration_min`, `fare_usd`, `tip_usd`, `rating` | What makes a ride expensive: distance, time of day, or the day of the week? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/rideshare_trips.csv")` |
| **used_cars** | One used-car listing | 1,000 x 8 | `make`, `body_type`, `fuel`, `year`, `mileage`, `price_usd`, `color`, `owners` | How fast do cars lose value as they age, and is it the same for every make and fuel type? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/used_cars.csv")` |
| **sneaker_resale** | One sneaker release, with its retail price and typical resale price | 500 x 8 | `brand`, `model`, `release_year`, `retail_usd`, `resale_usd`, `pairs_released_thousands`, `collab`, `colorway` | Which sneakers resell for more than they cost in the store, and what do those releases have in common? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/sneaker_resale.csv")` |
| **student_budgets** | One college student's typical month | 800 x 10 | `student_id`, `year_in_school`, `living`, `work_hours_week`, `monthly_income_usd`, `rent_usd`, `food_usd`, `transport_usd`, `entertainment_usd`, `savings_usd` | Where does a student's money actually go, and does living at home really save as much as people say? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/student_budgets.csv")` |
| **social_media_posts** | One post on a social app | 1,500 x 10 | `creator`, `followers_thousands`, `post_type`, `hour_posted`, `day_of_week`, `caption_words`, `hashtags`, `likes`, `comments`, `shares` | What gets a post more engagement: when you post it, what kind of post it is, or how many hashtags you use? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/social_media_posts.csv")` |
| **video_game_sales** | One video game release | 800 x 8 | `title`, `platform`, `genre`, `release_year`, `studio_size`, `critic_score`, `price_usd`, `units_sold_millions` | Do games with better reviews sell more, or does the size of the studio matter more? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/video_game_sales.csv")` |
| **fast_food_menu** | One menu item at a fast-food chain | 354 x 9 | `chain`, `item`, `category`, `calories`, `fat_g`, `carbs_g`, `protein_g`, `sodium_mg`, `price_usd` | Is the salad really the healthy choice, and which chain gives you the most calories per dollar? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/fast_food_menu.csv")` |

## Where it came from

Every file is rebuilt from its source by `python scripts/build.py` (needs `pandas` and `requests`). Data retrieved 2026-10-27. Run `pytest tests/` to check the files. `python scripts/synth.py` regenerates the synthetic sets from a fixed seed, so the files never change unless the script does.

Swaps from the original wish list, because there is no clearly open source for the original topic: NBA player stats, Spotify top songs, video game sales and fast food nutrition are all scraped or Kaggle-only with unclear licenses, so they became exoplanets, earthquakes, hurricanes and gapminder. US states income needs a Census API key, so `us_states` is population, births, deaths and migration instead.
