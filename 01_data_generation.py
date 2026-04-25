# Databricks notebook source
# Databricks Notebook 1: Data Generation

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

np.random.seed(42)

# COMMAND ----------

# Time range
weeks = pd.date_range(start="2023-01-01", periods=104, freq="W")

# Services in the city
services = ["metro", "bus", "private_transport", "water"]

# Base economics
base_prices = {
    "metro": 30,
    "bus": 10,
    "private_transport": 80,
    "water": 5
}

base_demand = {
    "metro": 500000,
    "bus": 800000,
    "private_transport": 600000,
    "water": 1e6
}

# Price elasticity
elasticity = {
    "metro": -1.3,
    "bus": -1.1,
    "private_transport": -0.8,
    "water": -0.4
}

# COMMAND ----------

def seasonality(t):
    return 1 + 0.1 * np.sin(2 * np.pi * t / 52)

def noise():
    return np.random.normal(0, 0.05)

def festival_flag(week):
    # Pongal, Diwali, year-end
    return 1 if week.month in [1, 10, 11, 12] else 0

def holiday_flag(week):
    return 1 if week.weekday() >= 5 else 0

# COMMAND ----------

rows = []

base_population = 1.1e7

for t, week in enumerate(weeks):

    # Temporal signals
    festival = festival_flag(week)
    holiday = holiday_flag(week)

    # Population drift (migration + festivals)
    pop_drift = 1 + np.random.uniform(-0.05, 0.1)
    if festival:
        pop_drift += 0.1

    population = base_population * pop_drift

    # Infrastructure dynamics
    congestion = 0.65 + np.random.uniform(-0.1, 0.1)
    congestion = min(1, max(0, congestion))

    signal_load = congestion * 0.6

    for s in services:

        # Price fluctuation
        price = base_prices[s] * (1 + np.random.uniform(-0.1, 0.1))

        # Demand via elasticity
        demand = base_demand[s] * ((price / base_prices[s]) ** elasticity[s])

        # Apply seasonality + noise
        demand *= seasonality(t)
        demand *= (1 + noise())

        # Traffic effects
        if s == "metro":
            demand *= (1 + congestion)
        elif s == "private_transport":
            demand *= (1 - congestion)

        # Festival / holiday behavior
        if festival:
            demand *= 1.2

        if holiday:
            if s in ["metro", "bus"]:
                demand *= 0.9
            if s == "private_transport":
                demand *= 1.1

        demand = max(1, demand)

        revenue = demand * price

        rows.append({
            "week": str(week),
            "service": s,
            "price": float(price),
            "demand": float(demand),
            "revenue": float(revenue),
            "population": float(population),
            "congestion": float(congestion),
            "signal_load": float(signal_load),
            "festival": int(festival),
            "holiday": int(holiday)
        })

df = pd.DataFrame(rows)

print("Sample Data:")
print(df.head())

print("\nShape:", df.shape)

# COMMAND ----------

spark_df = spark.createDataFrame(df)

spark_df.printSchema()

# COMMAND ----------

# Drop table if exists (clean rerun)
spark.sql("DROP TABLE IF EXISTS city_data")

# Save as managed Delta table (NO DBFS PATH)
spark_df.write.format("delta").mode("overwrite").saveAsTable("city_data")

# COMMAND ----------

# Quick check
spark.sql("SELECT * FROM city_data LIMIT 10").show()

# Count rows
spark.sql("SELECT COUNT(*) FROM city_data").show()

# COMMAND ----------


