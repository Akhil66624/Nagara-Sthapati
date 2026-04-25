# Databricks notebook source
# 02_model_training

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

df = spark.read.table("city_data").toPandas()

features = ["price", "population", "congestion", "signal_load", "festival", "holiday"]

X = df[features]
y = df["demand"]

with mlflow.start_run():

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # mlflow.sklearn.log_model(model, "model")

    mlflow.log_param("model_type", "RandomForest")
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        signature=mlflow.models.signature.infer_signature(X, y),
        registered_model_name="workspace.default.city_demand_forecast" # Use 3-level name for Unity Catalog     
    )

# COMMAND ----------


