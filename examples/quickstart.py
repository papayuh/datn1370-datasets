"""
One example chart per dataset. Run from the repo root:

    python examples/quickstart.py

PNG files are saved to examples/charts/. Copy any block below into your
own notebook as a starting point.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; remove this line in a notebook
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "examples" / "charts"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def finish(name, title):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=110)
    plt.close()
    print("saved", OUT / f"{name}.png")


# 1. Coffee shop sales -- bar chart of revenue by category and store
df = pd.read_csv(DATA / "coffee_shop_sales.csv", parse_dates=["date"])
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="category", y="total", hue="store", estimator="sum", errorbar=None)
plt.ylabel("Revenue ($)")
finish("coffee_shop_sales", "Coffee shop revenue by category and store")

# 2. Student performance -- scatter with regression line
df = pd.read_csv(DATA / "student_performance.csv")
plt.figure(figsize=(7, 5))
sns.regplot(data=df, x="study_hours_per_week", y="average_score", scatter_kws={"alpha": 0.25, "s": 12})
finish("student_performance", "Study hours vs. average score")

# 3. Weather -- monthly average high per city (line chart)
df = pd.read_csv(DATA / "city_weather_daily.csv", parse_dates=["date"])
monthly = df.groupby(["city", pd.Grouper(key="date", freq="MS")])["temp_high_f"].mean().reset_index()
plt.figure(figsize=(9, 5))
sns.lineplot(data=monthly, x="date", y="temp_high_f", hue="city")
plt.ylabel("Avg daily high (°F)")
finish("city_weather_daily", "Monthly average high temperature, 2022-2024")

# 4. Fitness tracker -- box plot of steps by fitness level
df = pd.read_csv(DATA / "fitness_tracker.csv", parse_dates=["date"])
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="fitness_level", y="steps", order=["Beginner", "Intermediate", "Advanced"])
finish("fitness_tracker", "Daily steps by fitness level")

# 5. Housing -- price vs. square footage colored by city
df = pd.read_csv(DATA / "housing_listings.csv")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sqft", y="price", hue="city", alpha=0.5, s=15)
plt.ylabel("Price ($)")
finish("housing_listings", "Listing price vs. square footage")

# 6. Movies -- average rating by genre (horizontal bar)
df = pd.read_csv(DATA / "movies.csv")
order = df.groupby("genre")["audience_rating"].mean().sort_values().index
plt.figure(figsize=(7, 5))
sns.barplot(data=df, y="genre", x="audience_rating", order=order, errorbar="sd")
finish("movies", "Average audience rating by genre")

# 7. E-commerce -- orders per month, stacked by device
df = pd.read_csv(DATA / "ecommerce_orders.csv", parse_dates=["order_date"])
counts = df.groupby(["order_month", "device"]).size().unstack(fill_value=0)
counts.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.xlabel("Month")
plt.ylabel("Orders")
finish("ecommerce_orders", "Orders per month by device (2024)")

# 8. Employee salaries -- salary distribution by department (violin)
df = pd.read_csv(DATA / "employee_salaries.csv")
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x="department", y="annual_salary", cut=0)
plt.xticks(rotation=30, ha="right")
finish("employee_salaries", "Salary distribution by department")

# 9. Music -- correlation heatmap of audio features
df = pd.read_csv(DATA / "music_tracks.csv")
features = ["energy", "danceability", "valence", "acousticness", "loudness_db", "tempo_bpm", "popularity"]
plt.figure(figsize=(7, 6))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
finish("music_tracks", "Correlation between audio features")

# 10. Cars -- combined MPG trend by fuel type
df = pd.read_csv(DATA / "car_fuel_economy.csv")
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="model_year", y="combined_mpg", hue="fuel_type", errorbar=None, marker="o")
plt.ylabel("Combined MPG (MPGe for electric)")
finish("car_fuel_economy", "Average fuel economy by model year")
