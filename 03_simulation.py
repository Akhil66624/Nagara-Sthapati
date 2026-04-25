# Databricks notebook source
# DBTITLE 1,Cell 1
# 03_simulation

import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("runs:/04f218bce72a4fe18c5594c8281148b1/model")

def simulate(policy):

    results = []

    for s in ["metro", "bus", "private_transport", "water"]:

        price = policy[f"{s}_price"]

        input_df = pd.DataFrame([{
            "price": price,
            "population": 1.1e7,
            "congestion": 0.65 + 0.3 * policy["fuel_tax"],
            "signal_load": 0.6,
            "festival": policy["festival"],
            "holiday": policy["holiday"]
        }])

        demand = model.predict(input_df)[0]

        revenue = demand * price

        results.append({
            "service": s,
            "demand": demand,
            "revenue": revenue
        })

    return pd.DataFrame(results)

# COMMAND ----------


