"""
Gold Layer: Aggregated metrics for smart city infrastructure analysis
Provides summarized data optimized for reasoning models and analytics
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.materialized_view(
    name="workspace.smart_city.gold_sensor_metrics_by_city_hour",
    comment="Hourly aggregated sensor metrics by city and sensor type for downstream reasoning models",
    cluster_by=["city_id", "sensor_type"]
)
def gold_sensor_metrics_by_city_hour():
    """
    Aggregate sensor data by city, sensor type, and hour.
    Uses batch read from streaming silver table for aggregation.
    
    Provides:
    - Traffic metrics: total and average vehicle counts
    - Energy metrics: total and average energy consumption
    - Waste metrics: average occupancy rates
    - Environmental metrics: average noise levels
    - Record counts per sensor type
    """
    return (
        spark.read.table("workspace.smart_city.silver_sensor_data")
            .withColumn("event_hour", F.date_trunc("hour", F.col("event_timestamp")))
            .groupBy("city_id", "sensor_type", "event_hour")
            .agg(
                # Traffic metrics (Trafik Sayacı)
                F.sum("vehicle_count").alias("total_vehicles"),
                F.avg("vehicle_count").alias("avg_vehicles"),
                F.max("vehicle_count").alias("max_vehicles"),
                
                # Energy metrics (Enerji Metre)
                F.sum("energy_kwh").alias("total_energy_kwh"),
                F.avg("energy_kwh").alias("avg_energy_kwh"),
                F.max("energy_kwh").alias("max_energy_kwh"),
                
                # Waste metrics (Atık Sensörü)
                F.avg("occupancy_rate").alias("avg_occupancy_rate"),
                F.max("occupancy_rate").alias("max_occupancy_rate"),
                
                # Environmental metrics (Çevre Sensörü)
                F.avg("noise_level").alias("avg_noise_level"),
                F.max("noise_level").alias("max_noise_level"),
                
                # Summary statistics
                F.count("*").alias("record_count"),
                F.countDistinct("sensor_id").alias("unique_sensors"),
                F.min("event_timestamp").alias("period_start"),
                F.max("event_timestamp").alias("period_end")
            )
    )


@dp.materialized_view(
    name="workspace.smart_city.gold_sensor_metrics_by_city_day",
    comment="Daily aggregated sensor metrics by city and sensor type",
    cluster_by=["city_id", "sensor_type"]
)
def gold_sensor_metrics_by_city_day():
    """
    Daily aggregation of sensor data by city and sensor type.
    Provides daily trends for analysis and reasoning models.
    """
    return (
        spark.read.table("workspace.smart_city.silver_sensor_data")
            .withColumn("event_date", F.to_date(F.col("event_timestamp")))
            .groupBy("city_id", "sensor_type", "event_date")
            .agg(
                # Traffic metrics
                F.sum("vehicle_count").alias("total_vehicles"),
                F.avg("vehicle_count").alias("avg_vehicles"),
                
                # Energy metrics
                F.sum("energy_kwh").alias("total_energy_kwh"),
                F.avg("energy_kwh").alias("avg_energy_kwh"),
                
                # Waste metrics
                F.avg("occupancy_rate").alias("avg_occupancy_rate"),
                
                # Environmental metrics
                F.avg("noise_level").alias("avg_noise_level"),
                
                # Summary
                F.count("*").alias("record_count"),
                F.countDistinct("sensor_id").alias("unique_sensors")
            )
    )
