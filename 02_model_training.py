# Databricks notebook source
# DBTITLE 1,Train model and log to MLflow
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

    mlflow.log_param("model_type", "RandomForest")
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        signature=mlflow.models.signature.infer_signature(X, y)
    )
    
    print(f"Model logged to MLflow experiment. Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------


