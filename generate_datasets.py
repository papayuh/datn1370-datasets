"""
Generate the 10 DATN 1370 practice datasets.

Run:  python generate_datasets.py
All files are written to ./data as tidy CSVs (one row per observation,
snake_case column names, no missing values, no duplicate rows).
Random seed is fixed so the output is reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 1370
rng = np.random.default_rng(SEED)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def save(df: pd.DataFrame, name: str) -> None:
    assert df.isna().sum().sum() == 0, f"{name} has missing values"
    assert not df.duplicated().any(), f"{name} has duplicate rows"
    df.to_csv(OUT / name, index=False)
    print(f"{name:32s} {len(df):>6,} rows x {df.shape[1]:>2} cols")


# --------------------------------------------------------------------------
# 1. Coffee shop sales  (transactions, dates, categories)
# --------------------------------------------------------------------------
def coffee_shop_sales(n=6000):
    stores = ["Downtown", "University", "Airport", "Suburb Mall"]
    store_w = [0.35, 0.3, 0.15, 0.2]
    menu = {
        "Coffee": [("Espresso", 2.75), ("Latte", 4.50), ("Cappuccino", 4.25),
                   ("Americano", 3.25), ("Cold Brew", 4.75)],
        "Tea": [("Green Tea", 3.00), ("Chai Latte", 4.50), ("Iced Tea", 3.25)],
        "Pastry": [("Croissant", 3.50), ("Blueberry Muffin", 3.25),
                   ("Cinnamon Roll", 4.00), ("Bagel", 2.50)],
        "Sandwich": [("Turkey Club", 8.50), ("Veggie Wrap", 7.75),
                     ("Ham & Cheese", 7.50)],
        "Smoothie": [("Berry Blast", 5.75), ("Mango Tango", 5.75),
                     ("Green Machine", 6.25)],
    }
    cat_w = {"Coffee": 0.45, "Tea": 0.15, "Pastry": 0.2, "Sandwich": 0.12, "Smoothie": 0.08}

    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, n), unit="D")
    hours = rng.choice(np.arange(6, 21), n, p=_hour_profile())
    cats = rng.choice(list(cat_w), n, p=list(cat_w.values()))
    products, prices = [], []
    for c in cats:
        item, price = menu[c][rng.integers(len(menu[c]))]
        products.append(item)
        prices.append(price)
    qty = rng.choice([1, 2, 3, 4], n, p=[0.6, 0.25, 0.1, 0.05])
    sizes = rng.choice(["Small", "Medium", "Large"], n, p=[0.3, 0.45, 0.25])
    size_add = np.select([sizes == "Medium", sizes == "Large"], [0.50, 1.00], 0.0)
    drink = np.isin(cats, ["Coffee", "Tea", "Smoothie"])
    unit_price = np.round(np.array(prices) + np.where(drink, size_add, 0.0), 2)
    df = pd.DataFrame({
        "transaction_id": np.arange(100001, 100001 + n),
        "date": dates,
        "hour": hours,
        "store": rng.choice(stores, n, p=store_w),
        "category": cats,
        "product": products,
        "size": np.where(drink, sizes, "Regular"),
        "quantity": qty,
        "unit_price": unit_price,
        "payment_method": rng.choice(["Card", "Cash", "Mobile"], n, p=[0.55, 0.2, 0.25]),
    })
    df["total"] = np.round(df.quantity * df.unit_price, 2)
    df["weekday"] = df.date.dt.day_name()
    df["month"] = df.date.dt.month_name()
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    df["transaction_id"] = np.arange(100001, 100001 + n)
    return df


def _hour_profile():
    w = np.array([2, 6, 10, 9, 6, 5, 7, 8, 6, 5, 5, 5, 4, 3, 2], float)
    return w / w.sum()


# --------------------------------------------------------------------------
# 2. Student performance
# --------------------------------------------------------------------------
def student_performance(n=3000):
    study = np.clip(rng.normal(10, 4, n), 0, 30).round(1)
    attendance = np.clip(rng.normal(88, 8, n), 40, 100).round(1)
    sleep = np.clip(rng.normal(7, 1.2, n), 3.5, 11).round(1)
    screen = np.clip(rng.normal(4, 1.8, n), 0, 12).round(1)
    base = 45 + 1.6 * study + 0.25 * attendance - 1.5 * screen + 2.0 * (sleep - 7)
    noise = rng.normal(0, 7, n)
    math = np.clip(base + noise + rng.normal(0, 5, n), 0, 100).round(0)
    reading = np.clip(base + noise + rng.normal(0, 5, n), 0, 100).round(0)
    science = np.clip(base + noise + rng.normal(0, 5, n), 0, 100).round(0)
    avg = (math + reading + science) / 3
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "gender": rng.choice(["Female", "Male", "Non-binary"], n, p=[0.49, 0.48, 0.03]),
        "grade_level": rng.choice([9, 10, 11, 12], n),
        "school_type": rng.choice(["Public", "Private", "Charter"], n, p=[0.7, 0.18, 0.12]),
        "lunch_program": rng.choice(["Standard", "Free/Reduced"], n, p=[0.62, 0.38]),
        "study_hours_per_week": study,
        "attendance_pct": attendance,
        "sleep_hours": sleep,
        "screen_time_hours": screen,
        "extracurricular": rng.choice(["No Activity", "Sports", "Music", "Clubs", "Multiple"], n,
                                      p=[0.3, 0.3, 0.15, 0.15, 0.1]),
        "math_score": math.astype(int),
        "reading_score": reading.astype(int),
        "science_score": science.astype(int),
    })
    df["average_score"] = avg.round(1)
    df["letter_grade"] = pd.cut(avg, [-1, 59.99, 69.99, 79.99, 89.99, 100],
                                labels=["F", "D", "C", "B", "A"]).astype(str)
    return df


# --------------------------------------------------------------------------
# 3. Daily weather for 5 cities, 3 years
# --------------------------------------------------------------------------
def city_weather_daily():
    cities = {  # mean annual temp (F), seasonal amplitude, rain prob, snow-capable
        "Austin": (69, 18, 0.22, False),
        "Chicago": (50, 25, 0.30, True),
        "Denver": (51, 22, 0.20, True),
        "Miami": (77, 8, 0.38, False),
        "Seattle": (53, 13, 0.45, False),
    }
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    doy = dates.dayofyear.values
    frames = []
    for city, (mean, amp, p_rain, snowy) in cities.items():
        season_curve = -np.cos((doy - 15) / 365.25 * 2 * np.pi)
        high = mean + 8 + amp * season_curve + rng.normal(0, 5, len(dates))
        low = high - rng.uniform(10, 22, len(dates))
        wet = rng.random(len(dates)) < p_rain
        precip = np.where(wet, rng.gamma(1.5, 0.3, len(dates)), 0.0).round(2)
        cond = np.where(wet & snowy & (high < 36), "Snow",
                        np.where(wet, "Rain",
                                 rng.choice(["Sunny", "Partly Cloudy", "Cloudy"], len(dates),
                                            p=[0.5, 0.3, 0.2])))
        frames.append(pd.DataFrame({
            "date": dates,
            "city": city,
            "temp_high_f": high.round(1),
            "temp_low_f": low.round(1),
            "precipitation_in": precip,
            "humidity_pct": np.clip(rng.normal(55 + 25 * wet, 12), 10, 100).round(0).astype(int),
            "wind_mph": np.clip(rng.gamma(3, 2.5, len(dates)), 0, 45).round(1),
            "condition": cond,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["season"] = df.month.map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                                 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                                 9: "Fall", 10: "Fall", 11: "Fall"})
    return df.sort_values(["date", "city"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Fitness tracker (100 users x 60 days)
# --------------------------------------------------------------------------
def fitness_tracker(n_users=100, n_days=60):
    users = pd.DataFrame({
        "user_id": [f"U{str(i).zfill(3)}" for i in range(1, n_users + 1)],
        "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n_users,
                                p=[0.15, 0.3, 0.25, 0.18, 0.12]),
        "gender": rng.choice(["Female", "Male"], n_users),
        "fitness_level": rng.choice(["Beginner", "Intermediate", "Advanced"], n_users,
                                    p=[0.35, 0.45, 0.2]),
    })
    lvl_steps = {"Beginner": 5500, "Intermediate": 8500, "Advanced": 12000}
    dates = pd.date_range("2024-03-01", periods=n_days)
    df = users.merge(pd.DataFrame({"date": dates}), how="cross")
    n = len(df)
    weekend = df.date.dt.dayofweek >= 5
    base = df.fitness_level.map(lvl_steps).values
    steps = np.clip(rng.normal(base * np.where(weekend, 0.85, 1.0), 2500), 500, 30000)
    df["steps"] = steps.round(0).astype(int)
    df["distance_km"] = (df.steps * rng.uniform(0.00070, 0.00080, n)).round(2)
    df["active_minutes"] = np.clip(df.steps / 110 + rng.normal(0, 10, n), 5, 300).round(0).astype(int)
    df["calories_burned"] = (1500 + df.steps * 0.045 + rng.normal(0, 120, n)).round(0).astype(int)
    df["sleep_hours"] = np.clip(rng.normal(7.1 + 0.4 * weekend, 1.1), 3, 12).round(1)
    df["resting_heart_rate"] = np.clip(
        rng.normal(72 - 5 * (df.fitness_level == "Advanced") + 3 * (df.fitness_level == "Beginner"), 6),
        45, 100).round(0).astype(int)
    df["workout_type"] = rng.choice(["Rest", "Walking", "Running", "Cycling", "Strength", "Yoga"],
                                    n, p=[0.3, 0.25, 0.15, 0.1, 0.12, 0.08])
    df["day_of_week"] = df.date.dt.day_name()
    return df


# --------------------------------------------------------------------------
# 5. Housing listings
# --------------------------------------------------------------------------
def housing_listings(n=4000):
    cities = {"Austin": 230, "Denver": 260, "Phoenix": 200, "Nashville": 215,
              "Raleigh": 190, "Portland": 270, "Tampa": 205, "Columbus": 160}
    city = rng.choice(list(cities), n)
    ppsf = np.array([cities[c] for c in city]) * rng.normal(1, 0.12, n)
    bedrooms = rng.choice([1, 2, 3, 4, 5], n, p=[0.07, 0.2, 0.4, 0.26, 0.07])
    sqft = np.clip(rng.normal(600 + 520 * bedrooms, 250), 450, 5500).round(-1).astype(int)
    year_built = rng.integers(1950, 2025, n)
    home_type = rng.choice(["Single Family", "Townhouse", "Condo"], n, p=[0.6, 0.2, 0.2])
    garage = rng.random(n) < np.where(home_type == "Condo", 0.3, 0.8)
    price = ppsf * sqft * (1 + 0.004 * (year_built - 1990)) * np.where(garage, 1.05, 1.0)
    df = pd.DataFrame({
        "listing_id": np.arange(50001, 50001 + n),
        "city": city,
        "home_type": home_type,
        "bedrooms": bedrooms,
        "bathrooms": np.clip(bedrooms - rng.choice([0, 0.5, 1, 1.5], n, p=[0.4, 0.3, 0.2, 0.1]), 1, 5),
        "sqft": sqft,
        "lot_acres": np.where(home_type == "Condo", 0.0,
                              np.clip(rng.gamma(2, 0.12, n), 0.05, 2.5).round(2)),
        "year_built": year_built,
        "has_garage": garage,
        "has_pool": rng.random(n) < np.where(np.isin(city, ["Phoenix", "Tampa", "Austin"]), 0.3, 0.08),
        "price": (price / 1000).round(0).astype(int) * 1000,
        "days_on_market": np.clip(rng.gamma(2, 15, n), 1, 180).round(0).astype(int),
        "listing_month": rng.choice(pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m"), n),
    })
    df["price_per_sqft"] = (df.price / df.sqft).round(2)
    df["home_age"] = 2024 - df.year_built
    return df


# --------------------------------------------------------------------------
# 6. Movies
# --------------------------------------------------------------------------
def movies(n=3000):
    adj = ["Silent", "Broken", "Golden", "Last", "Hidden", "Electric", "Lost", "Crimson",
           "Midnight", "Infinite", "Burning", "Frozen", "Secret", "Wild", "Quiet", "Distant"]
    noun = ["Horizon", "Kingdom", "Echo", "Promise", "Garden", "Signal", "Harbor", "Empire",
            "Shadow", "River", "Legacy", "Storm", "Mirror", "Journey", "Code", "Island"]
    genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Animation", "Thriller", "Documentary"]
    g_w = [0.18, 0.16, 0.2, 0.1, 0.1, 0.08, 0.07, 0.08, 0.03]
    g_budget = {"Action": 110, "Comedy": 40, "Drama": 30, "Horror": 15, "Sci-Fi": 120,
                "Romance": 28, "Animation": 95, "Thriller": 45, "Documentary": 5}
    g_rating = {"Action": 6.3, "Comedy": 6.1, "Drama": 6.9, "Horror": 5.6, "Sci-Fi": 6.5,
                "Romance": 6.2, "Animation": 6.8, "Thriller": 6.4, "Documentary": 7.2}
    genre = rng.choice(genres, n, p=g_w)
    year = rng.integers(1990, 2025, n)
    budget = np.clip(np.array([g_budget[g] for g in genre]) * rng.lognormal(0, 0.5, n), 0.5, 350)
    rating = np.clip(np.array([g_rating[g] for g in genre]) + rng.normal(0, 0.9, n), 1.5, 9.6)
    multiplier = np.clip(rng.lognormal(0.3, 0.8, n) * (0.6 + 0.12 * (rating - 5)), 0.05, 12)
    box = np.minimum(budget * multiplier, 2200)
    titles = [f"{a} {b}" for a, b in zip(rng.choice(adj, n), rng.choice(noun, n))]
    # make titles unique by appending a numeral when repeated
    seen, uniq = {}, []
    for t in titles:
        seen[t] = seen.get(t, 0) + 1
        uniq.append(t if seen[t] == 1 else f"{t} {seen[t]}")
    df = pd.DataFrame({
        "movie_id": np.arange(1, n + 1),
        "title": uniq,
        "release_year": year,
        "genre": genre,
        "mpaa_rating": rng.choice(["G", "PG", "PG-13", "R"], n, p=[0.05, 0.2, 0.45, 0.3]),
        "runtime_min": np.clip(rng.normal(108, 17, n), 70, 200).round(0).astype(int),
        "studio": rng.choice(["Paramount Pictures", "Universal", "Warner Bros.", "Sony Pictures",
                              "Disney", "Lionsgate", "A24", "Independent"], n,
                             p=[0.13, 0.15, 0.15, 0.12, 0.14, 0.08, 0.05, 0.18]),
        "budget_musd": budget.round(1),
        "box_office_musd": box.round(1),
        "audience_rating": rating.round(1),
        "num_votes": (rng.lognormal(10, 1.3, n)).round(0).astype(int),
    })
    df["profit_musd"] = (df.box_office_musd - df.budget_musd).round(1)
    df["decade"] = (df.release_year // 10 * 10).astype(str) + "s"
    return df


# --------------------------------------------------------------------------
# 7. E-commerce orders
# --------------------------------------------------------------------------
def ecommerce_orders(n=6000):
    catalog = {
        "Electronics": [("Wireless Earbuds", 59.99), ("Bluetooth Speaker", 44.99),
                        ("Phone Case", 14.99), ("USB-C Cable", 9.99), ("Smart Watch", 199.99)],
        "Clothing": [("T-Shirt", 19.99), ("Jeans", 49.99), ("Hoodie", 39.99),
                     ("Sneakers", 79.99), ("Jacket", 89.99)],
        "Home & Kitchen": [("Coffee Maker", 69.99), ("Cutting Board", 24.99),
                           ("Throw Blanket", 34.99), ("LED Desk Lamp", 29.99)],
        "Books": [("Paperback Novel", 12.99), ("Hardcover Nonfiction", 24.99), ("Cookbook", 29.99)],
        "Sports": [("Yoga Mat", 24.99), ("Water Bottle", 17.99), ("Resistance Bands", 15.99),
                   ("Running Shorts", 27.99)],
        "Beauty": [("Face Serum", 32.99), ("Shampoo", 11.99), ("Lip Balm Set", 8.99)],
    }
    cat_w = [0.22, 0.28, 0.15, 0.12, 0.13, 0.10]
    cats = rng.choice(list(catalog), n, p=cat_w)
    prods, prices = [], []
    for c in cats:
        p, pr = catalog[c][rng.integers(len(catalog[c]))]
        prods.append(p)
        prices.append(pr)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, n), unit="D")
    # busier in Nov/Dec
    boost = rng.random(n) < 0.18
    dates = np.where(boost, pd.to_datetime("2024-11-01") + pd.to_timedelta(rng.integers(0, 61, n), unit="D"), dates)
    qty = rng.choice([1, 2, 3, 4, 5], n, p=[0.55, 0.25, 0.11, 0.06, 0.03])
    discount = rng.choice([0, 5, 10, 15, 20, 25], n, p=[0.5, 0.12, 0.15, 0.1, 0.08, 0.05])
    ship = rng.choice(["Standard", "Express", "Overnight"], n, p=[0.65, 0.27, 0.08])
    ship_cost = pd.Series(ship).map({"Standard": 4.99, "Express": 9.99, "Overnight": 19.99}).values
    df = pd.DataFrame({
        "order_id": np.arange(900001, 900001 + n),
        "order_date": pd.to_datetime(dates),
        "customer_id": [f"C{v}" for v in rng.integers(1000, 2500, n)],
        "customer_region": rng.choice(["Northeast", "Southeast", "Midwest", "Southwest", "West"], n,
                                      p=[0.2, 0.22, 0.2, 0.15, 0.23]),
        "customer_age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n,
                                         p=[0.18, 0.32, 0.24, 0.15, 0.11]),
        "device": rng.choice(["Mobile", "Desktop", "Tablet"], n, p=[0.58, 0.34, 0.08]),
        "category": cats,
        "product": prods,
        "quantity": qty,
        "unit_price": prices,
        "discount_pct": discount,
        "shipping_method": ship,
        "shipping_cost": ship_cost,
    })
    df["subtotal"] = (df.quantity * df.unit_price * (1 - df.discount_pct / 100)).round(2)
    df["order_total"] = (df.subtotal + df.shipping_cost).round(2)
    df["returned"] = rng.random(n) < np.where(df.category == "Clothing", 0.14, 0.05)
    df["order_month"] = df.order_date.dt.month
    df = df.sort_values("order_date").reset_index(drop=True)
    df["order_id"] = np.arange(900001, 900001 + n)
    return df


# --------------------------------------------------------------------------
# 8. Employee salaries
# --------------------------------------------------------------------------
def employee_salaries(n=3500):
    depts = {"Engineering": 95, "Sales": 62, "Marketing": 66, "Finance": 78,
             "Human Resources": 60, "Operations": 58, "Customer Support": 46, "Product": 92}
    dept = rng.choice(list(depts), n, p=[0.28, 0.17, 0.1, 0.1, 0.06, 0.12, 0.1, 0.07])
    level = rng.choice(["Junior", "Mid", "Senior", "Lead", "Manager", "Director"], n,
                       p=[0.25, 0.3, 0.22, 0.1, 0.09, 0.04])
    lvl_mult = {"Junior": 0.75, "Mid": 1.0, "Senior": 1.3, "Lead": 1.5, "Manager": 1.65, "Director": 2.1}
    lvl_exp = {"Junior": 1.5, "Mid": 5, "Senior": 9, "Lead": 12, "Manager": 13, "Director": 18}
    edu = rng.choice(["High School", "Bachelor's", "Master's", "PhD"], n, p=[0.12, 0.55, 0.28, 0.05])
    edu_mult = {"High School": 0.92, "Bachelor's": 1.0, "Master's": 1.08, "PhD": 1.15}
    city = rng.choice(["New York", "San Francisco", "Austin", "Chicago", "Denver", "Atlanta", "Remote"], n,
                      p=[0.18, 0.15, 0.15, 0.12, 0.1, 0.1, 0.2])
    city_mult = {"New York": 1.2, "San Francisco": 1.3, "Austin": 1.0, "Chicago": 1.05,
                 "Denver": 1.0, "Atlanta": 0.95, "Remote": 1.0}
    exp = np.clip(np.array([lvl_exp[l] for l in level]) + rng.normal(0, 2.5, n), 0, 40).round(1)
    salary = (np.array([depts[d] for d in dept]) * 1000
              * np.array([lvl_mult[l] for l in level])
              * np.array([edu_mult[e] for e in edu])
              * np.array([city_mult[c] for c in city])
              * (1 + 0.01 * exp) * rng.normal(1, 0.07, n))
    perf = rng.choice([1, 2, 3, 4, 5], n, p=[0.04, 0.12, 0.45, 0.3, 0.09])
    df = pd.DataFrame({
        "employee_id": np.arange(10001, 10001 + n),
        "department": dept,
        "job_level": level,
        "education": edu,
        "gender": rng.choice(["Female", "Male"], n, p=[0.46, 0.54]),
        "age": np.clip(22 + exp + rng.normal(3, 3, n), 21, 67).round(0).astype(int),
        "years_experience": exp,
        "office_city": city,
        "work_mode": np.where(city == "Remote", "Remote",
                              rng.choice(["On-site", "Hybrid"], n, p=[0.4, 0.6])),
        "performance_rating": perf,
        "annual_salary": (salary / 500).round(0).astype(int) * 500,
    })
    df["bonus"] = (df.annual_salary * (0.02 * perf + rng.uniform(-0.01, 0.01, n))).clip(0).round(-2).astype(int)
    df["years_at_company"] = np.clip(np.minimum(exp, rng.gamma(2, 2, n)), 0, 40).round(1)
    return df


# --------------------------------------------------------------------------
# 9. Music tracks
# --------------------------------------------------------------------------
def music_tracks(n=4000):
    genres = ["Pop", "Hip-Hop", "Rock", "Electronic", "R&B", "Country", "Jazz", "Classical", "Latin", "Indie"]
    g_w = [0.2, 0.17, 0.14, 0.12, 0.08, 0.08, 0.05, 0.04, 0.06, 0.06]
    g_tempo = {"Pop": 118, "Hip-Hop": 95, "Rock": 128, "Electronic": 126, "R&B": 90,
               "Country": 112, "Jazz": 110, "Classical": 80, "Latin": 100, "Indie": 115}
    g_energy = {"Pop": 0.68, "Hip-Hop": 0.65, "Rock": 0.78, "Electronic": 0.8, "R&B": 0.55,
                "Country": 0.6, "Jazz": 0.4, "Classical": 0.25, "Latin": 0.75, "Indie": 0.55}
    g_dur = {"Pop": 200, "Hip-Hop": 190, "Rock": 230, "Electronic": 240, "R&B": 215,
             "Country": 205, "Jazz": 300, "Classical": 360, "Latin": 210, "Indie": 220}
    first = ["Neon", "Velvet", "Paper", "Glass", "Summer", "Winter", "Ocean", "Desert", "City", "Golden",
             "Lonely", "Electric", "Silver", "Midnight", "Sunday", "Faded"]
    second = ["Lights", "Hearts", "Dreams", "Roads", "Skies", "Nights", "Waves", "Fire", "Rain", "Echoes",
              "Stars", "Streets", "Love", "Memories", "Horizon", "Shadows"]
    artists = [f"{a} {b}" for a in ["The", "DJ", "Little", "Big", "Young", "Saint", "Lady", "Lord"]
               for b in ["Foxes", "Nova", "Harbor", "Atlas", "Marigold", "Cassette", "Orbit", "Juniper",
                         "Wren", "Pixel", "Sable", "Monarch"]]
    genre = rng.choice(genres, n, p=g_w)
    year = rng.integers(1980, 2025, n)
    energy = np.clip(np.array([g_energy[g] for g in genre]) + rng.normal(0, 0.13, n), 0.02, 0.99)
    dance = np.clip(0.3 + 0.5 * energy + rng.normal(0, 0.12, n), 0.05, 0.99)
    valence = np.clip(0.2 + 0.5 * dance + rng.normal(0, 0.18, n), 0.02, 0.99)
    acoustic = np.clip(1 - energy + rng.normal(0, 0.15, n), 0.0, 1.0)
    popularity = np.clip(40 + 25 * dance + 0.4 * (year - 2000) + rng.normal(0, 14, n), 0, 100)
    df = pd.DataFrame({
        "track_id": np.arange(1, n + 1),
        "title": [f"{a} {b}" for a, b in zip(rng.choice(first, n), rng.choice(second, n))],
        "artist": rng.choice(artists, n),
        "genre": genre,
        "release_year": year,
        "duration_sec": np.clip(np.array([g_dur[g] for g in genre]) + rng.normal(0, 35, n), 60, 900).round(0).astype(int),
        "tempo_bpm": np.clip(np.array([g_tempo[g] for g in genre]) + rng.normal(0, 14, n), 50, 210).round(0).astype(int),
        "energy": energy.round(3),
        "danceability": dance.round(3),
        "valence": valence.round(3),
        "acousticness": acoustic.round(3),
        "loudness_db": np.clip(-20 + 14 * energy + rng.normal(0, 2, n), -40, 0).round(1),
        "explicit": rng.random(n) < np.where(genre == "Hip-Hop", 0.55, 0.1),
        "popularity": popularity.round(0).astype(int),
    })
    df["streams_millions"] = (np.exp(df.popularity / 18) * rng.lognormal(0, 0.4, n)).round(2)
    df["decade"] = (df.release_year // 10 * 10).astype(str) + "s"
    return df


# --------------------------------------------------------------------------
# 10. Car fuel economy
# --------------------------------------------------------------------------
def car_fuel_economy(n=3200):
    makes = {"Toyota": 1.0, "Honda": 1.0, "Ford": 0.9, "Chevrolet": 0.88, "Nissan": 0.95,
             "Hyundai": 1.02, "Kia": 1.02, "Subaru": 0.92, "BMW": 0.9, "Mercedes-Benz": 0.88,
             "Volkswagen": 0.97, "Mazda": 1.0, "Jeep": 0.82, "Tesla": 1.0}
    make = rng.choice(list(makes), n)
    vclass = rng.choice(["Compact", "Midsize Sedan", "SUV", "Pickup Truck", "Minivan", "Sports Car", "Hatchback"],
                        n, p=[0.18, 0.2, 0.28, 0.14, 0.06, 0.06, 0.08])
    class_mpg = {"Compact": 32, "Midsize Sedan": 29, "SUV": 24, "Pickup Truck": 19,
                 "Minivan": 22, "Sports Car": 21, "Hatchback": 31}
    class_cyl = {"Compact": [4], "Midsize Sedan": [4, 6], "SUV": [4, 6, 8], "Pickup Truck": [6, 8],
                 "Minivan": [6], "Sports Car": [6, 8], "Hatchback": [4]}
    fuel = np.where(make == "Tesla", "Electric",
                    rng.choice(["Gasoline", "Hybrid", "Diesel"], n, p=[0.78, 0.17, 0.05]))
    cyl = np.array([rng.choice(class_cyl[c]) for c in vclass])
    cyl = np.where(fuel == "Electric", 0, cyl)
    disp = np.where(fuel == "Electric", 0.0, np.clip(0.55 * cyl + rng.normal(0, 0.3, n), 1.0, 7.0).round(1))
    year = rng.integers(2010, 2025, n)
    base = np.array([class_mpg[c] for c in vclass]) * np.array([makes[m] for m in make])
    base = base * (1 + 0.012 * (year - 2010)) - 1.2 * (cyl - 4)
    fuel_mult = np.select([fuel == "Hybrid", fuel == "Diesel", fuel == "Electric"], [1.55, 1.15, 3.6], 1.0)
    combined = np.clip(base * fuel_mult * rng.normal(1, 0.06, n), 12, 140)
    city = combined * rng.uniform(0.85, 0.93, n)
    hwy = combined * rng.uniform(1.08, 1.2, n)
    msrp = (np.array([{"Compact": 24, "Midsize Sedan": 30, "SUV": 38, "Pickup Truck": 42,
                       "Minivan": 36, "Sports Car": 55, "Hatchback": 25}[c] for c in vclass]) * 1000
            * np.where(np.isin(make, ["BMW", "Mercedes-Benz", "Tesla"]), 1.6, 1.0)
            * (1 + 0.025 * (year - 2010)) * rng.normal(1, 0.1, n))
    df = pd.DataFrame({
        "vehicle_id": np.arange(1, n + 1),
        "make": make,
        "model_year": year,
        "vehicle_class": vclass,
        "fuel_type": fuel,
        "cylinders": cyl,
        "engine_liters": disp,
        "transmission": np.where(fuel == "Electric", "Automatic",
                                 rng.choice(["Automatic", "Manual", "CVT"], n, p=[0.72, 0.1, 0.18])),
        "drivetrain": rng.choice(["FWD", "RWD", "AWD", "4WD"], n, p=[0.45, 0.15, 0.3, 0.1]),
        "city_mpg": city.round(0).astype(int),
        "highway_mpg": hwy.round(0).astype(int),
        "combined_mpg": combined.round(0).astype(int),
        "msrp_usd": (msrp / 100).round(0).astype(int) * 100,
    })
    df["co2_g_per_mile"] = np.where(df.fuel_type == "Electric", 0,
                                    (8887 / df.combined_mpg).round(0)).astype(int)
    gas_cost = 15000 / df.combined_mpg * np.where(df.fuel_type == "Diesel", 3.9, 3.5)
    df["annual_fuel_cost_usd"] = np.where(df.fuel_type == "Electric", 650, gas_cost.round(-1)).astype(int)
    return df


if __name__ == "__main__":
    save(coffee_shop_sales(), "coffee_shop_sales.csv")
    save(student_performance(), "student_performance.csv")
    save(city_weather_daily(), "city_weather_daily.csv")
    save(fitness_tracker(), "fitness_tracker.csv")
    save(housing_listings(), "housing_listings.csv")
    save(movies(), "movies.csv")
    save(ecommerce_orders(), "ecommerce_orders.csv")
    save(employee_salaries(), "employee_salaries.csv")
    save(music_tracks(), "music_tracks.csv")
    save(car_fuel_economy(), "car_fuel_economy.csv")
