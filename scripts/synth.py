"""Make the made-up datasets in data/synthetic/. Run: python scripts/synth.py [name ...]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "synthetic"
DAYS = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])


def pick(rng, items, weights, n):
    w = np.array(weights, dtype=float)
    return rng.choice(items, size=n, p=w / w.sum())


def by(table, keys, default=0.0):
    return np.array([table.get(k, default) for k in keys], dtype=float)


def coffee_shop_sales(rng):
    # one day at one shop of a small coffee chain
    dates = pd.date_range("2025-01-01", "2025-12-31")
    n = len(dates)
    doy = dates.dayofyear.values
    m, d = dates.month.values, dates.day.values
    weekend = dates.dayofweek.values >= 5
    temp = (78 - 16 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 6, n)).clip(35, 103).round().astype(int)
    share = np.clip(0.10 + 0.009 * (temp - 45), 0.08, 0.80)
    frames = []
    for shop, base, wk in [("Downtown", 220, 0.65), ("Campus", 180, 0.70), ("Riverside", 140, 1.25)]:
        lam = base * np.where(weekend, wk, 1.0)
        if shop == "Campus":
            summer = (m == 6) | (m == 7) | ((m == 8) & (d <= 15))
            winter = ((m == 12) & (d >= 15)) | ((m == 1) & (d <= 12))
            lam = lam * np.where(summer, 0.6, 1.0) * np.where(winter, 0.5, 1.0)
        if shop == "Riverside":
            lam = lam * (1 + 0.5 * doy / 365)
        total = rng.poisson(lam)
        iced = rng.binomial(total, share)
        hot = total - iced
        pastries = rng.poisson(0.35 * total * np.where(weekend, 1.2, 1.0))
        revenue = (hot * 4.25 + iced * 5.25 + pastries * 3.50 + rng.normal(0, 15, n)).clip(min=0).round(2)
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"), "shop": shop, "hot_drinks": hot, "iced_drinks": iced,
            "pastries": pastries, "revenue_usd": revenue, "temp_high_f": temp,
        }))
    return pd.concat(frames).sort_values("date", kind="stable").reset_index(drop=True)


def gym_members(rng):
    # one member of a gym who joined in 2025
    n = 1200
    plan = pick(rng, ["student", "basic", "plus"], [30, 45, 25], n)
    plus, student = plan == "plus", plan == "student"
    age = np.select(
        [student, plan == "basic"],
        [rng.normal(21, 2.5, n).clip(18, 30), rng.normal(36, 11, n).clip(18, 75)],
        rng.normal(42, 12, n).clip(18, 75),
    ).round().astype(int)
    w = np.full(12, 7.0)
    w[0], w[1], w[8] = 22, 10, 12
    month = pick(rng, np.arange(1, 13), w, n)
    jan = month == 1
    classes = rng.poisson(np.select([plus, student], [4.0, 2.0], 1.5))
    v1 = np.maximum(rng.poisson(9 + 3 * plus + 1 * student + 2 * jan), 1)
    z = -0.8 + 0.10 * v1 + 0.25 * classes + 0.5 * plus - 1.3 * jan
    active = rng.random(n) < 1 / (1 + np.exp(-z))
    v6 = np.where(active, rng.poisson(v1 * np.clip(0.45 + 0.08 * classes, 0.3, 1.1)), 0)
    return pd.DataFrame({
        "member_id": [f"M{i:04d}" for i in range(1, n + 1)], "age": age, "plan": plan, "join_month": month,
        "visits_month_1": v1, "visits_month_6": v6, "classes_per_month": classes, "still_active": active,
    })


def streaming_songs(rng):
    # one song on a streaming service
    n = 1000
    genres = ["pop", "hip hop", "rock", "country", "electronic", "r&b", "indie", "latin"]
    firsts = "Nova Jules Remy Sasha Indigo Marlo Tatum Rowan Zadie Ellis Kai Dove Lux Wren Ione Beck Sol Ari Mika Juno".split()
    lasts = ["Vale", "Hartwell", "Okafor", "Reyes-Lin", "Castellano", "Brightwater", "Moreau", "Takeda",
             "Fairbanks", "Quill", "Ashby", "Delacroix", "Nkemelu", "Sorensen", "Pike"]
    plurals = ["Lanterns", "Night Owls", "Paper Boats", "Satellites", "Streetlights", "Hummingbirds", "Static Kids",
               "Rooftops", "Wanderers", "Postcards", "Fog Machines", "Weekenders", "Daydreamers", "Side Streets", "Comets"]
    adjs = ("Midnight Golden Broken Electric Velvet Paper Neon Hollow Quiet Wild Crimson Faded Static Honest "
            "Sunlit Restless Frozen Lonely Candy Rusty").split()
    nouns = ("River Highway Satellite Mirror Garden Thunder Window Ocean Letter Fever Horizon Lantern Compass "
             "Shadow Rooftop Engine Postcard Harbor Ember Avenue").split()
    verbs = "Call Hold Find Leave Keep Carry Follow Wake Dance Miss".split()
    adverbs = "Tonight Slowly Again Forever Anyway Someday Softly Twice".split()
    people = [f"{a} {b}" for a in firsts for b in lasts]
    bands = ["The " + p for p in plurals] + [f"{a} {p}" for a in adjs for p in plurals]
    artists = list(rng.choice(people, 75, replace=False)) + list(rng.choice(bands, 75, replace=False))
    artist_genre = rng.choice(genres, 150)
    weight = rng.lognormal(0, 0.5, 150)
    a = rng.choice(150, size=n, p=weight / weight.sum())
    genre = artist_genre[a]
    years = np.arange(2000, 2026)
    year = pick(rng, years, 1 + 0.15 * (years - 2000), n)
    tempo_mean = {"pop": 118, "hip hop": 92, "rock": 128, "country": 110, "electronic": 126, "r&b": 85, "indie": 115, "latin": 100}
    duration = (rng.normal(250 - 2.2 * (year - 2000), 30) + by({"hip hop": -10, "rock": 20, "electronic": 15, "pop": -8}, genre))
    tempo = (by(tempo_mean, genre) + rng.normal(0, 12, n)).clip(60, 190).round().astype(int)
    energy_off = {"electronic": 0.15, "rock": 0.10, "r&b": -0.10, "country": -0.05}
    energy = (0.2 + 0.0035 * tempo + by(energy_off, genre) + rng.normal(0, 0.08, n)).clip(0.05, 0.99).round(2)
    genre_factor = {"pop": 1.6, "hip hop": 1.5, "country": 0.7, "indie": 0.5}
    streams = (rng.lognormal(np.log(20), 1.1, n) * by(genre_factor, genre, 1.0) * weight[a]
               * np.interp(year, [2010, 2021, 2025], [1.0, 1.8, 1.2]))
    hits = rng.choice(n, 8, replace=False)
    streams[hits] = rng.uniform(1500, 4500, 8)
    seen, titles = set(), []
    while len(titles) < n:
        k = rng.integers(5)
        if k == 0:
            t = f"{rng.choice(adjs)} {rng.choice(nouns)}"
        elif k == 1:
            n1, n2 = rng.choice(nouns, 2, replace=False)
            t = f"{n1} in the {n2}"
        elif k == 2:
            t = f"{rng.choice(verbs)} Me {rng.choice(adverbs)}"
        elif k == 3:
            t = str(rng.choice(nouns))
        else:
            a1, a2 = rng.choice(adjs, 2, replace=False)
            t = f"{a1} {a2} {rng.choice(nouns)}"
        free = [t + s for s in ["", " (Remix)", " II", " III", " IV"] if t + s not in seen]
        if free:
            seen.add(free[0])
            titles.append(free[0])
    df = pd.DataFrame({
        "title": titles, "artist": [artists[i] for i in a], "genre": genre, "release_year": year,
        "duration_sec": duration.clip(90, 420).round().astype(int), "tempo_bpm": tempo, "energy": energy,
        "streams_millions": streams.clip(min=0.1).round(1),
    })
    return df.sort_values(["title", "artist"]).reset_index(drop=True)


def rideshare_trips(rng):
    # one trip with a rideshare app in one city
    n = 1500
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 365, n)), unit="D")
    dow = dates.dayofweek.values
    weekday = dow < 5
    weekend_night = (dow == 4) | (dow == 5)
    # hour of day: commute peaks, evening peak, or any time
    u = rng.random(n)
    evening = u < np.where(weekend_night, 0.50, 0.30)
    anytime = (u >= 0.85) & ~evening
    peak = np.where(rng.random(n) < 0.5, 8, 17)
    hour = np.where(evening, rng.normal(21, 1.5, n), rng.normal(peak, 1.5, n))
    hour = np.where(anytime, rng.integers(0, 24, n), hour).round().astype(int) % 24
    zones = ["Downtown", "Midtown", "University", "Airport", "Heights", "Eastside", "Suburbs"]
    zone = pick(rng, zones, [25, 18, 15, 12, 12, 10, 8], n)
    median = by({"Airport": 14, "Suburbs": 8}, zone, 4.0)
    sigma = by({"Airport": 0.25, "Suburbs": 0.4}, zone, 0.6)
    distance = (median * np.exp(sigma * rng.normal(size=n))).clip(0.5, 40).round(1)
    rush = weekday & (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18)))
    night = (hour >= 23) | (hour <= 5)
    speed = np.select([rush, night], [13.0, 32.0], 24.0) * rng.lognormal(0, 0.15, n)
    duration = np.maximum((distance / speed * 60 + 3).round(), 3).astype(int)
    surge_time = weekend_night & ((hour >= 21) | (hour <= 1))
    surge = np.where(surge_time, rng.uniform(1.3, 1.9, n), 1.0)
    fare = ((2.50 + 1.10 * distance + 0.28 * duration) * surge + rng.normal(0, 0.8, n)).clip(min=5).round(2)
    stars = [5, 4, 3, 2, 1]
    rating = np.where(speed < 12,
                      rng.choice(stars, n, p=[0.45, 0.30, 0.15, 0.05, 0.05]),
                      rng.choice(stars, n, p=[0.72, 0.18, 0.06, 0.02, 0.02]))
    tip_chance = np.select([rating == 5, rating == 4], [0.65, 0.45], 0.20)
    late = (hour >= 22) | (hour <= 2)
    pct = 0.15 + 0.05 * (zone == "Airport") + 0.05 * late + rng.normal(0, 0.06, n)
    tip = np.clip(fare * pct, 0.5, 0.4 * fare).round(2)
    tip = np.where(rng.random(n) < tip_chance, tip, 0.0)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "hour": hour, "day_of_week": DAYS[dow], "pickup_zone": zone,
        "distance_miles": distance, "duration_min": duration, "fare_usd": fare, "tip_usd": tip, "rating": rating,
    })


def used_cars(rng):
    # one used-car listing; the car makes are made up
    n = 988
    makes = ["Aurora", "Kestrel", "Marlow", "Halcyon", "Vireo", "Tandem"]
    make_w = [15, 20, 20, 12, 18, 15]
    years = np.arange(2008, 2026)
    year = pick(rng, years, np.interp(years, [2008, 2019, 2025], [1, 4, 2]), n)
    body = pick(rng, ["sedan", "suv", "truck", "hatchback", "coupe"], [32, 30, 15, 13, 10], n)
    fuel = pick(rng, ["gas", "hybrid", "electric"], [70, 18, 12], n)
    old = (fuel == "electric") & (year < 2015)
    fuel[old] = pick(rng, ["gas", "hybrid"], [70, 18], old.sum())
    make = pick(rng, makes, make_w, n)
    age = 2026 - year
    mileage = (11500 * age * rng.lognormal(0, 0.3, n)).clip(500, 300000).astype(int)
    owners = np.clip(1 + rng.poisson(age / 7), 1, 5)
    base = by({"sedan": 28000, "suv": 38000, "truck": 45000, "hatchback": 24000, "coupe": 36000}, body)
    factor = by({"Aurora": 1.25, "Kestrel": 0.85, "Halcyon": 1.10, "Vireo": 0.90}, make, 1.0)
    msrp = base * factor + by({"hybrid": 4000, "electric": 12000}, fuel)
    k = np.where(fuel == "electric", 0.17, np.where(make == "Halcyon", 0.06, 0.10))
    price = msrp * np.exp(-k * age) * np.exp(-1.2e-6 * mileage) * (1 - 0.02 * (owners - 1)) * rng.lognormal(0, 0.12, n)
    regular = pd.DataFrame({"make": make, "body_type": body, "fuel": fuel, "year": year, "mileage": mileage,
                            "price_usd": price.clip(min=1500).astype(int), "owners": owners})
    # 12 old classic cars
    c = 12
    classic = pd.DataFrame({
        "make": pick(rng, makes, make_w, c), "body_type": rng.choice(["coupe", "truck"], c), "fuel": "gas",
        "year": rng.integers(1965, 1980, c), "mileage": rng.integers(60000, 150001, c),
        "price_usd": rng.integers(25000, 90001, c), "owners": rng.integers(3, 6, c),
    })
    df = pd.concat([regular, classic], ignore_index=True)
    df["color"] = pick(rng, ["white", "black", "gray", "silver", "blue", "red", "green", "orange"],
                       [24, 20, 18, 12, 10, 9, 4, 3], len(df))
    df = df[["make", "body_type", "fuel", "year", "mileage", "price_usd", "color", "owners"]]
    return df.sort_values(["make", "year"], kind="stable").reset_index(drop=True)


def sneaker_resale(rng):
    # one sneaker release with its retail price and typical resale price
    n = 500
    families = {"Stride": ["Orbit", "Pulse", "Vector"], "Zephyr": ["Gale", "Drift", "Cirrus"],
                "Kickwell": ["Court", "Flex", "Rally"], "Fleetfoot": ["Dash", "Sprint", "Marathon"],
                "Northline": ["Summit", "Ridge", "Trail"]}
    colors = ["black/white", "triple white", "navy/gold", "cherry", "sage", "cream", "royal blue", "grape",
              "olive", "sunset orange", "slate", "bone"]
    brand = rng.choice(list(families), n)
    model = [f"{rng.choice(families[b])} {rng.integers(1, 13)}" for b in brand]
    year = rng.integers(2015, 2026, n)
    collab = rng.random(n) < 0.18
    median = np.where(collab, 15, 60)
    sigma = np.where(collab, 0.7, 0.9)
    pairs = (median * np.exp(sigma * rng.normal(size=n))).clip(2, 800).round(1)
    base = by({"Stride": 150, "Zephyr": 130, "Kickwell": 110, "Fleetfoot": 120, "Northline": 170}, brand)
    retail = (base + 40 * collab + 3 * (year - 2015) + rng.normal(0, 12, n)) / 5
    retail = retail.round().astype(int) * 5
    log_ratio = 0.25 - 0.45 * np.log(pairs / 60) + 0.6 * collab - 0.06 * (year - 2015) + rng.normal(0, 0.35, n)
    ratio = np.exp(log_ratio)
    spikes = rng.choice(n, 6, replace=False)
    ratio[spikes] *= rng.uniform(5, 12, 6)
    df = pd.DataFrame({
        "brand": brand, "model": model, "release_year": year, "retail_usd": retail,
        "resale_usd": np.maximum(retail * ratio, 40).astype(int), "pairs_released_thousands": pairs,
        "collab": collab, "colorway": rng.choice(colors, n),
    })
    return df.sort_values(["brand", "release_year"], kind="stable").reset_index(drop=True)


def student_budgets(rng):
    # one college student's typical month
    n = 800
    year = pick(rng, ["freshman", "sophomore", "junior", "senior"], [30, 27, 23, 20], n)
    homes = ["with family", "on campus", "off campus"]
    mix = {"freshman": [45, 40, 15], "sophomore": [40, 30, 30], "junior": [35, 15, 50], "senior": [30, 10, 60]}
    living = np.array([pick(rng, homes, mix[y], 1)[0] for y in year])
    fam, camp = living == "with family", living == "on campus"
    upper = (year == "junior") | (year == "senior")
    works = rng.random(n) >= np.where(upper, 0.25, 0.30)
    hours = np.where(works, (rng.normal(18, 8, n) + 3 * upper).clip(4, 40).round(), 0).astype(int)
    support = np.select([fam, camp], [350, 450], 550)
    income = np.maximum(hours * 15.5 * 4.3 + support + rng.normal(0, 150, n), 200).round().astype(int)
    rent = np.select(
        [fam, camp],
        [np.where(rng.random(n) < 0.70, 0, rng.uniform(100, 400, n)), rng.normal(1100, 120, n)],
        rng.normal(850, 180, n),
    ).clip(0, 2000).round().astype(int)
    food = np.select([fam, camp], [rng.normal(180, 50, n), rng.normal(380, 50, n)], rng.normal(320, 80, n))
    food = (food + 3.5 * hours).clip(min=60).round().astype(int)
    transport = np.select([fam, camp], [rng.normal(170, 60, n), rng.normal(45, 25, n)], rng.normal(130, 50, n))
    transport = transport.clip(min=0).round().astype(int)
    fun = (rng.normal(130, 55, n) + 1.5 * hours).clip(min=10).round().astype(int)
    return pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)], "year_in_school": year, "living": living,
        "work_hours_week": hours, "monthly_income_usd": income, "rent_usd": rent, "food_usd": food,
        "transport_usd": transport, "entertainment_usd": fun,
        "savings_usd": income - (rent + food + transport + fun),
    })


def social_media_posts(rng):
    # one post on a social app
    n = 1500
    adjs = "pixel cozy salty tiny loud moody sunny lazy crispy hyper retro velvet dusty minty bold".split()
    nouns = "pancakes plants pixels sneakers maps noodles comets teacups bikes cats notebooks waves lamps puzzles beats".split()
    handles = rng.choice([f"{a}_{b}" for a in adjs for b in nouns], 60, replace=False)
    followers = (25 * np.exp(1.3 * rng.normal(size=60))).clip(0.5, 5000).round(1)
    c = rng.integers(0, 60, n)
    kind = pick(rng, ["video", "photo", "text"], [45, 40, 15], n)
    w = np.array([1] * 7 + [3] * 2 + [5] * 9 + [7] * 5 + [2])
    hour = pick(rng, np.arange(24), w, n)
    text, video = kind == "text", kind == "video"
    words = np.maximum(rng.poisson(np.where(text, 35, 18)), 1)
    tags = rng.poisson(4, n).clip(0, 15)
    log_likes = (np.log(followers[c] * 1000 * 0.03) + by({"video": 0.35, "text": -0.5}, kind)
                 + 0.3 * ((hour >= 18) & (hour <= 22)) + 0.08 * np.minimum(tags, 5)
                 - 0.10 * np.maximum(tags - 5, 0) + rng.normal(0, 0.6, n))
    likes = np.maximum(np.exp(log_likes), 1)
    viral = rng.choice(n, 5, replace=False)
    likes[viral] *= rng.uniform(20, 40, 5)
    likes = likes.astype(int)
    df = pd.DataFrame({
        "creator": handles[c], "followers_thousands": followers[c], "post_type": kind, "hour_posted": hour,
        "day_of_week": rng.choice(DAYS, n), "caption_words": words, "hashtags": tags, "likes": likes,
        "comments": rng.poisson(likes * 0.025 * np.where(text, 2.2, 1.0)),
        "shares": rng.poisson(likes * 0.012 * np.where(video, 3.0, 1.0)),
    })
    return df.sort_values("creator", kind="stable").reset_index(drop=True)


def video_game_sales(rng):
    # one video game release; the studios, platforms and titles are made up
    n = 800
    adjs = "Ember Hollow Iron Silent Last Neon Crimson Frozen Lost Rogue Astral Savage Tiny Infinite Quantum".split()
    nouns = ("Citadel Vanguard Driftwood Starfall Reckoning Horizon Tidewater Bastion Wasteland Kingdom Garden "
             "Harvest Circuit Kart League Dungeon Frontier Signal Oracle Expanse").split()
    subs = ["Awakening", "Rebirth", "Shadows Rising", "Final Lap", "Legends", "Uprising", "Deep Dive", "Reloaded",
            "Origins", "Overdrive"]
    genres = ["action", "rpg", "sports", "shooter", "puzzle", "racing", "strategy", "simulation", "platformer", "horror"]
    plats = ["PC", "Pocket", "Nova", "Orbit", "Cloud"]
    plat_w = {"indie": [40, 20, 15, 15, 10], "mid": [25, 15, 25, 25, 10], "major": [15, 10, 35, 35, 5]}
    year = rng.integers(2005, 2026, n)
    u = rng.random(n)
    studio = np.where(u < 0.25 + 0.35 * (year - 2005) / 20, "indie", np.where(rng.random(n) < 0.55, "mid", "major"))
    indie, mid, major = studio == "indie", studio == "mid", studio == "major"
    platform, genre = [], []
    for i in range(n):
        p = pick(rng, plats, plat_w[studio[i]], 1)[0]
        if p == "Cloud" and year[i] < 2020:
            p = "PC"
        g = np.ones(10)
        if p == "Pocket":
            g[[4, 8]] = 3
        if p in ("Nova", "Orbit"):
            g[[2, 3]] = 2
        platform.append(p)
        genre.append(pick(rng, genres, g, 1)[0])
    genre = np.array(genre)
    score = (rng.normal(72, np.where(indie, 14, 11)) + 4 * major).clip(30, 98).round().astype(int)
    price = np.select([indie, mid], [rng.normal(18, 6, n).clip(5, 35), rng.normal(40, 8, n).clip(20, 60)],
                      np.where(year < 2020, 59.99, 69.99))
    price = np.where(major, price, np.round(price) - 0.01).round(2)
    log_units = (-2.0 + 0.035 * (score - 70) + by({"indie": -1.2, "major": 1.3}, studio)
                 + 0.3 * (major & (genre == "sports")) - 0.4 * (genre == "puzzle") - 0.2 * (genre == "horror")
                 + rng.normal(0, 0.9, n))
    units = np.exp(log_units)
    a = rng.choice(np.where(indie)[0], 10, replace=False)
    units[a] *= rng.uniform(30, 60, 10)
    b = rng.choice(np.where(major & (score < 60))[0], 6, replace=False)
    units[b] *= rng.uniform(4, 8, 6)
    seen, titles = set(), []
    while len(titles) < n:
        k = rng.integers(5)
        if k == 0:
            t = f"{rng.choice(adjs)} {rng.choice(nouns)}"
        elif k == 1:
            n1, n2 = rng.choice(nouns, 2, replace=False)
            t = f"{n1} of {n2}"
        elif k == 2:
            t = f"{rng.choice(nouns)}: {rng.choice(subs)}"
        elif k == 3:
            t = f"{rng.choice(nouns)} {rng.choice(['II', 'III', 'IV', 'V'])}"
        else:
            t = f"{rng.choice(nouns)}bound"
        if t not in seen:
            seen.add(t)
            titles.append(t)
    df = pd.DataFrame({
        "title": titles, "platform": platform, "genre": genre, "release_year": year, "studio_size": studio,
        "critic_score": score, "price_usd": price, "units_sold_millions": units.clip(min=0.01).round(2),
    })
    return df.sort_values("title").reset_index(drop=True)


def fast_food_menu(rng):
    # one menu item at a fast-food chain
    chains = ["Burger Barn", "Taco Tempo", "Cluck House", "Pie Society", "Wrap Republic", "Noodle Nook"]
    foods = {"Burger Barn": ["Burger", "Melt", "Stack"], "Taco Tempo": ["Taco", "Burrito", "Quesadilla"],
             "Cluck House": ["Chicken Sandwich", "Tenders", "Wings"], "Pie Society": ["Pizza", "Calzone", "Flatbread"],
             "Wrap Republic": ["Wrap", "Bowl", "Pita"], "Noodle Nook": ["Noodles", "Ramen", "Stir Fry"]}
    mains = ("Classic Spicy Smoky Crispy Loaded Double Garden Ranch Honey Sesame Chipotle Teriyaki BBQ Buffalo "
             "Southwest Cheddar Jalapeno Truffle").split()
    names = {
        "side": ["Fries", "Curly Fries", "Onion Rings", "Tots", "Chips", "Rice", "Slaw", "Mac and Cheese", "Corn",
                 "Breadsticks", "Edamame", "Dumplings", "Garlic Knots", "Mozzarella Sticks"],
        "salad": [f"{s} Salad" for s in "Caesar Garden Southwest Cobb Chicken Greek Harvest Taco Noodle".split()],
        "dessert": ["Brownie", "Cookie", "Sundae", "Shake", "Churro", "Pie Slice", "Cheesecake", "Cinnamon Twist", "Mochi"],
        "drink": ["Cola", "Diet Cola", "Lemonade", "Iced Tea", "Sweet Tea", "Orange Soda", "Root Beer", "Coffee",
                  "Hot Chocolate", "Smoothie", "Bubble Tea", "Horchata"],
        "breakfast": ["Egg Burrito", "Breakfast Sandwich", "Hash Browns", "Pancakes", "Oatmeal", "Breakfast Bowl",
                      "Biscuit", "French Toast Sticks", "Yogurt Parfait", "Croissant"],
    }
    count = {"main": 18, "side": 10, "salad": 6, "dessert": 7, "drink": 10, "breakfast": 8}
    cal_base = {"main": 650, "side": 320, "salad": 420, "dessert": 420, "drink": 220, "breakfast": 480}
    cal_chain = {"Burger Barn": 1.20, "Cluck House": 1.10, "Pie Society": 1.15, "Wrap Republic": 0.85, "Noodle Nook": 0.95}
    macros = {"main": (0.42, 0.38, 0.20), "side": (0.45, 0.48, 0.07), "salad": (0.50, 0.30, 0.20),
              "dessert": (0.35, 0.60, 0.05), "drink": (0.05, 0.92, 0.03), "breakfast": (0.45, 0.38, 0.17)}
    price_base = {"main": 7.50, "side": 3.00, "salad": 8.50, "dessert": 3.50, "drink": 2.20, "breakfast": 5.00}
    price_chain = {"Wrap Republic": 1.30, "Pie Society": 1.10, "Noodle Nook": 1.05}
    rows = []
    for chain in chains:
        for cat, k in count.items():
            if cat == "main":
                pool = [f"{a} {f}" for a in mains for f in foods[chain]]
            else:
                pool = names[cat]
            for item in rng.choice(pool, k, replace=False):
                rows.append((chain, str(item), cat))
    df = pd.DataFrame(rows, columns=["chain", "item", "category"])
    n = len(df)
    cat = df.category
    calories = by(cal_base, cat) * by(cal_chain, df.chain, 1.0) * np.where((df.chain == "Wrap Republic") & (cat == "salad"), 1.5, 1.0)
    calories = (calories * rng.lognormal(0, 0.25, n)).clip(40, 1800).round().astype(int)
    frac = np.array([macros[c] for c in cat]) + rng.normal(0, 0.04, (n, 3))
    frac = frac.clip(min=0.01)
    frac = frac / frac.sum(axis=1, keepdims=True)
    sodium = calories * 1.4 * by({"drink": 0.10, "dessert": 0.40}, cat, 1.0) * np.where(df.chain == "Taco Tempo", 1.5, 1.0)
    price = by(price_base, cat) * by(price_chain, df.chain, 1.0) * rng.lognormal(0, 0.15, n)
    df["calories"] = calories
    df["fat_g"] = (calories * frac[:, 0] / 9).round(1)
    df["carbs_g"] = (calories * frac[:, 1] / 4).round(1)
    df["protein_g"] = (calories * frac[:, 2] / 4).round(1)
    df["sodium_mg"] = (sodium * rng.lognormal(0, 0.2, n)).astype(int)
    df["price_usd"] = (np.floor(price) + rng.choice([0.29, 0.49, 0.79, 0.99], n)).round(2)
    return df.sort_values("chain", kind="stable").reset_index(drop=True)


DATASETS = [coffee_shop_sales, gym_members, streaming_songs, rideshare_trips, used_cars,
            sneaker_resale, student_budgets, social_media_posts, video_game_sales, fast_food_menu]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1:]
    for i, fn in enumerate(DATASETS):
        if want and fn.__name__ not in want:
            continue
        df = fn(np.random.default_rng([1370, i]))
        df.to_csv(OUT / f"{fn.__name__}.csv", index=False)
        print(f"{fn.__name__}: {len(df)} rows x {df.shape[1]} cols")
