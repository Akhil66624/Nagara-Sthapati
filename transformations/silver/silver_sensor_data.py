"""
Silver Layer: Cleaned and standardized smart city sensor data
- Renames Turkish columns to English
- Applies data quality checks
- Ensures data integrity
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.table(
    name="workspace.smart_city.silver_sensor_data",
    comment="Cleaned smart city sensor data with English column names and quality checks",
    cluster_by=["sensor_type", "city_id"]
)
@dp.expect("valid_sensor_id", "sensor_id IS NOT NULL")
@dp.expect("valid_timestamp", "event_timestamp IS NOT NULL")
@dp.expect("valid_coordinates", "latitude IS NOT NULL AND longitude IS NOT NULL")
@dp.expect_or_drop("valid_latitude", "latitude BETWEEN -90 AND 90")
@dp.expect_or_drop("valid_longitude", "longitude BETWEEN -180 AND 180")
@dp.expect_or_drop("valid_sensor_type", "sensor_type IN ('Trafik Sayacı', 'Enerji Metre', 'Atık Sensörü', 'Çevre Sensörü')")
def silver_sensor_data():
    """
    Read from bronze layer and transform:
    - Rename Turkish columns to English
    - Cast data types appropriately
    - Add processing metadata
    """
    return (
        spark.readStream.table("workspace.smart_city.bronze_sensor_data")
            .select(
                # City and Sensor identification
                F.col("`Şehir ID'si/Adı`").alias("city_id"),
                F.col("`Sensör ID'si/Adı`").alias("sensor_id"),
                
                # Location
                F.col("Enlem").cast("double").alias("latitude"),
                F.col("Boylam").cast("double").alias("longitude"),
                
                # Timestamp
                F.col("`Tarih/Zaman`").cast("timestamp").alias("event_timestamp"),
                
                # Sensor metadata
                F.col("`Sensör Tipi`").alias("sensor_type"),
                F.col("`Sokak Tipi`").alias("street_type"),
                F.col("`Yakındaki Hizmetler`").alias("nearby_services"),
                
                # Metrics - different sensors have different non-null values
                F.col("`Araç Sayısı`").cast("integer").alias("vehicle_count"),
                F.col("kWh").cast("double").alias("energy_kwh"),
                F.col("`Doluluk Oranı`").cast("double").alias("occupancy_rate"),
                F.col("`Gürültü Seviyesi`").cast("double").alias("noise_level"),
                
                # Processing metadata
                F.current_timestamp().alias("processed_at")
            )
    )
