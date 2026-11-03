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
| **houston_weather** | One day at Hobby Airport, 2022-2024 | 1,092 x 7 | `date`, `temp_high_f`, `temp_low_f`, `rain_in`, `wind_mph`, `season`, `rain_status` | How do daily high temperatures and rainfall vary by season? | [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily), station USW00012918. US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/houston_weather.csv")` |
| **earthquakes** | One magnitude 5+ earthquake in 2024 | 1,507 x 8 | `date`, `latitude`, `longitude`, `depth_km`, `magnitude`, `place`, `depth_band`, `magnitude_band` | Are deeper earthquakes bigger or smaller? Where do the biggest ones cluster? | [USGS](https://earthquake.usgs.gov/fdsnws/event/1/). US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/earthquakes.csv")` |
| **exoplanets** | One confirmed planet outside our solar system (random 1,500 with complete measurements) | 1,500 x 8 | `planet`, `discovery_method`, `year_discovered`, `orbit_days`, `radius_earths`, `mass_earths`, `distance_parsecs`, `discovery_decade` | Which discovery method finds which kind of planet? How did that change by year? | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/), operated by Caltech/IPAC for NASA. Follow its [acknowledgment policy](https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html); not a blanket public-domain license | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/exoplanets.csv")` |
| **gapminder** | One country in one year (every 5 years, 1955-2005) | 682 x 6 | `year`, `country`, `population`, `life_expect`, `fertility`, `continent` | Did every country get healthier at the same pace? Which ones broke the pattern? | [Gapminder Foundation](https://www.gapminder.org/data/) via vega-datasets. CC-BY 4.0, credit Gapminder | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/gapminder.csv")` |
| **us_states** | One state (or DC) in one year, 2011-2024 | 714 x 8 | `state`, `year`, `population`, `births`, `deaths`, `net_migration`, `region`, `migration_direction` | Which states are growing fastest, and is it births or people moving in? | [US Census Bureau Population Estimates](https://www.census.gov/programs-surveys/popest.html). US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/us_states.csv")` |
| **movies** | One movie, 1970-2013 | 1,776 x 7 | `title`, `year`, `bechdel_result`, `budget_2013_usd`, `domestic_gross_2013_usd`, `worldwide_gross_2013_usd`, `decade` | Do movies that pass the Bechdel test make less money? Does budget predict gross? | [FiveThirtyEight](https://github.com/fivethirtyeight/data/tree/master/bechdel) (data from bechdeltest.com). CC-BY 4.0, credit FiveThirtyEight | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/movies.csv")` |
| **hurricanes** | One named Atlantic storm, 1980-2025 | 611 x 7 | `name`, `year`, `month`, `days_active`, `max_wind_mph`, `min_pressure_mb`, `made_landfall` | Are storms getting stronger over time? Which months are the worst? | [NOAA National Hurricane Center HURDAT2](https://www.nhc.noaa.gov/data/#hurdat). US government, public domain | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/hurricanes.csv")` |

## Notes

- `houston_weather`: days with a missing reading were dropped (1,092 of 1,096 days kept). Temperatures converted to F, rain to inches, wind to mph. `season` uses Dec-Feb winter, Mar-May spring, Jun-Aug summer, Sep-Nov fall. `rain_status` is `rain` when `rain_in > 0`, otherwise `no measurable rain` (trace rain can round to zero).
- `earthquakes`: `depth_band` is `shallow` below 70 km, `intermediate` from 70 to below 300 km, and `deep` at 300 km or deeper. `magnitude_band` groups magnitudes below 6, from 6 to below 7, and 7+. Magnitude is unitless; latitude/longitude are degrees.
- `exoplanets`: only planets with every measurement filled in, then restricted to the four most frequent methods **among those complete records**, then randomly sampled down to 1,500. The sample includes Transit, Radial Velocity, Transit Timing Variations and Imaging; the last two groups are small, so use counts or scatterplots rather than a violin plot for them. `radius_earths` and `mass_earths` are multiples of Earth's radius and mass. `discovery_decade` groups discovery years. Measurements span very wide ranges; a log scale or a clearly labeled filter can help. Acknowledge the NASA Exoplanet Archive, operated by Caltech/IPAC under contract with NASA, when using these data; see the linked policy for citation details.
- `gapminder`: the source `cluster` column was dropped; `continent` uses six geographic continents. Central America and the Caribbean are in North America; Georgia and Turkey are in Asia. `life_expect` is life expectancy in years; `fertility` is children per woman. Countries repeat across years: filter to one year for a country comparison, or group by both country and year.
- `us_states`: 50 states + DC, Puerto Rico left out. 2010 left out because its births/deaths only cover Apr-Dec. Vintage 2010-2019 and 2020-2024 estimates are stacked, so there can be a small jump at 2020. `region` uses the four Census regions (DC is in the South). `migration_direction` is `net inflow`, `net outflow`, or `no change` according to the sign of `net_migration`; negative migration means more people left than arrived. Counts are people per year, except `population`, which is a population estimate. States repeat across years: filter to one year before comparing state totals.
- `movies`: dollar columns are in 2013 dollars. Movies missing budget or gross were dropped. Some titles repeat because of remakes: use **title and year together**, not title alone, to identify or group movies. `decade` groups release years.
- `hurricanes`: unnamed storms, numbered depressions (such as `Ten` and `Twenty-One`), and storms before 1980 dropped. This removes 23 numbered records from the earlier 634-row file. `max_wind_mph` is the highest wind and `min_pressure_mb` the lowest pressure over the storm's life; they need not occur at the same observation. `month` is the starting month (1-12), `days_active` counts calendar dates touched, and `made_landfall` is True/False. Storm names repeat: use name and year together.

## Synthetic (made-up) datasets

These are generated by `scripts/synth.py`. Every number and name in them is invented. Nothing here describes a real business, person, product or place, so never cite them as facts. They are here because the real menu lacks everyday consumer topics. Same rules as above: clean, no missing values, snake_case columns.

| dataset | one row is | rows x cols | columns | starter question | source + license | load it |
|---|---|---|---|---|---|---|
| **coffee_shop_sales** | One day at one shop of a small coffee chain, 2025 | 1,095 x 8 | `date`, `shop`, `hot_drinks`, `iced_drinks`, `pastries`, `revenue_usd`, `temp_high_f`, `day_of_week` | Which shop does best on weekends, and does hot weather change what people order? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/coffee_shop_sales.csv")` |
| **gym_members** | One member of a gym who joined in 2025 | 1,200 x 8 | `member_id`, `age`, `plan`, `join_month`, `visits_month_1`, `visits_month_6`, `classes_per_month`, `still_active` | Do people who join in January stick around as long as everyone else? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/gym_members.csv")` |
| **streaming_songs** | One song on a streaming service | 1,000 x 9 | `title`, `artist`, `genre`, `release_year`, `duration_sec`, `tempo_bpm`, `energy`, `streams_millions`, `release_decade` | Have songs gotten shorter over the years, and does that matter for how many streams they get? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/streaming_songs.csv")` |
| **rideshare_trips** | One trip with a rideshare app in one city, 2025 | 1,500 x 9 | `date`, `hour`, `day_of_week`, `pickup_zone`, `distance_miles`, `duration_min`, `fare_usd`, `tip_usd`, `rating` | What makes a ride expensive: distance, time of day, or the day of the week? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/rideshare_trips.csv")` |
| **used_cars** | One used-car listing | 1,000 x 8 | `make`, `body_type`, `fuel`, `year`, `mileage`, `price_usd`, `color`, `owners` | How fast do cars lose value as they age, and is it the same for every make and fuel type? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/used_cars.csv")` |
| **sneaker_resale** | One sneaker release, with its retail price and typical resale price | 500 x 8 | `brand`, `model`, `release_year`, `retail_usd`, `resale_usd`, `pairs_released_thousands`, `collab`, `colorway` | Which sneakers resell for more than they cost in the store, and what do those releases have in common? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/sneaker_resale.csv")` |
| **student_budgets** | One college student's typical month | 800 x 10 | `student_id`, `year_in_school`, `living`, `work_hours_week`, `monthly_income_usd`, `rent_usd`, `food_usd`, `transport_usd`, `entertainment_usd`, `savings_usd` | Where does a student's money actually go, and does living at home really save as much as people say? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/student_budgets.csv")` |
| **social_media_posts** | One post on a social app | 1,500 x 10 | `creator`, `followers_thousands`, `post_type`, `hour_posted`, `day_of_week`, `caption_words`, `hashtags`, `likes`, `comments`, `shares` | What gets a post more engagement: when you post it, what kind of post it is, or how many hashtags you use? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/social_media_posts.csv")` |
| **video_game_sales** | One video game release | 800 x 8 | `title`, `platform`, `genre`, `release_year`, `studio_size`, `critic_score`, `price_usd`, `units_sold_millions` | Do games with better reviews sell more, or does the size of the studio matter more? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/video_game_sales.csv")` |
| **fast_food_menu** | One menu item at a fast-food chain | 354 x 9 | `chain`, `item`, `category`, `calories`, `fat_g`, `carbs_g`, `protein_g`, `sodium_mg`, `price_usd` | Is the salad really the healthy choice, and which chain gives you the most calories per dollar? | Made up. Generated by scripts/synth.py. Use freely (CC0) | `pd.read_csv("https://raw.githubusercontent.com/papayuh/datn1370-datasets/main/data/synthetic/fast_food_menu.csv")` |

### Synthetic data notes

- Deliberate outliers are included: viral social posts, hit songs, classic used cars, hype sneakers, surprise indie game hits, and some low-scoring major-studio games with high sales. These are invented observations, not errors to delete automatically. Outliers can stretch chart axes; a box plot, log scale for positive values, or a clearly labeled filtered view can help.
- `student_budgets.savings_usd` is monthly income minus the four expense columns. Negative savings means a monthly shortfall, not bad data. Keep the sign; a log scale does not work for zero or negative values.
- `used_cars.mileage` is miles; `gym_members.age` is years. `streaming_songs.energy` is a unitless 0-1 score; `video_game_sales.critic_score` is a 0-100 score; rideshare `rating` is 1-5 stars. Drink/pastry counts, visits, owners, words, hashtags and engagement counts are counts, not dollars. `_thousands` and `_millions` columns are already scaled; do not multiply unless converting units.
- Coffee dates repeat once per shop. `day_of_week` uses Mon-Sun; sort weekdays explicitly if order matters. Song `release_decade` groups release years. `gym_members.visits_month_6` is zero for inactive members; `still_active` and sneaker `collab` are True/False.

## Chart picker

Two different chart types that fit each file. These are options, not required questions or findings. A box plot compares a numeric column across groups; a scatterplot uses two numeric columns. A bar chart usually needs counts or a grouped mean/sum first. For stacked bars, start with `pd.crosstab(df["category_1"], df["category_2"]).plot.bar(stacked=True)` and replace the category names with columns from your file. No file requires heavy cleanup.

| dataset | rows | key columns | two chart types |
|---|---:|---|---|
| penguins | 333 | `species`, `sex`, `flipper_length_mm`, `body_mass_g` | box plot; scatterplot |
| houston_weather | 1,092 | `season`, `rain_status`, `temp_high_f`, `rain_in` | box plot; histogram |
| earthquakes | 1,507 | `depth_band`, `magnitude_band`, `depth_km`, `magnitude` | bar; scatterplot |
| exoplanets | 1,500 | `discovery_method`, `discovery_decade`, `radius_earths`, `mass_earths` | bar; scatterplot |
| gapminder | 682 | `continent`, `year`, `life_expect`, `fertility` | box plot; scatterplot |
| us_states | 714 | `region`, `migration_direction`, `births`, `deaths` | box plot; scatterplot |
| movies | 1,776 | `decade`, `bechdel_result`, `budget_2013_usd`, `worldwide_gross_2013_usd` | stacked bar; scatterplot |
| hurricanes | 611 | `month`, `made_landfall`, `max_wind_mph`, `min_pressure_mb` | box plot; scatterplot |
| coffee_shop_sales | 1,095 | `shop`, `day_of_week`, `temp_high_f`, `revenue_usd` | box plot; scatterplot |
| gym_members | 1,200 | `plan`, `still_active`, `visits_month_1`, `visits_month_6` | stacked bar; scatterplot |
| streaming_songs | 1,000 | `genre`, `release_decade`, `duration_sec`, `streams_millions` | box plot; scatterplot |
| rideshare_trips | 1,500 | `pickup_zone`, `day_of_week`, `distance_miles`, `fare_usd` | box plot; scatterplot |
| used_cars | 1,000 | `body_type`, `fuel`, `mileage`, `price_usd` | box plot; scatterplot |
| sneaker_resale | 500 | `brand`, `collab`, `retail_usd`, `resale_usd` | box plot; scatterplot |
| student_budgets | 800 | `living`, `year_in_school`, `monthly_income_usd`, `savings_usd` | violin plot; scatterplot |
| social_media_posts | 1,500 | `post_type`, `day_of_week`, `followers_thousands`, `likes` | box plot; scatterplot |
| video_game_sales | 800 | `genre`, `studio_size`, `critic_score`, `units_sold_millions` | box plot; scatterplot |
| fast_food_menu | 354 | `category`, `chain`, `calories`, `sodium_mg` | box plot; scatterplot |

All dates are ISO `YYYY-MM-DD` strings. To use date operations, run `df["date"] = pd.to_datetime(df["date"])`. Existing column names and raw URLs are unchanged; the grouping columns above were added for easier charting. Units are in column names where provided by this project; the older short names are explained in the notes above.

## Where it came from

Real files are rebuilt from their sources by `python scripts/build.py` (needs `pandas` and `requests`). Initially retrieved 2026-10-27; files with added groupings and hurricanes refreshed 2026-11-03. Run `pytest tests/` to check all 18 files, their chart-ready categories, numeric ranges, dates, encoding, and synthetic reproducibility. `python scripts/synth.py` regenerates the synthetic sets from a fixed seed (needs `numpy` and `pandas`), so the files never change unless the script does. Either script accepts dataset names to rebuild only those files.

Code is MIT-licensed; see [LICENSE](LICENSE). Data are excluded from that code license: each file keeps the source license or acknowledgment policy in its menu row above. Synthetic data are dedicated under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

Swaps from the original wish list, because there is no clearly open source for the original topic: NBA player stats, Spotify top songs, video game sales and fast food nutrition are all scraped or Kaggle-only with unclear licenses, so they became exoplanets, earthquakes, hurricanes and gapminder. US states income needs a Census API key, so `us_states` is population, births, deaths and migration instead.
