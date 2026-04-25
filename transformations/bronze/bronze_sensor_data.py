"""
Bronze Layer: Raw ingestion of smart city sensor data from CSV using Auto Loader
Monitors the volume path for new CSV files and streams them continuously
"""

from pyspark import pipelines as dp

@dp.table(
    name="workspace.smart_city.bronze_sensor_data",
    comment="Raw smart city sensor data ingested from CSV files using Auto Loader",
    table_properties={
        "delta.columnMapping.mode": "name"
    }
)
def bronze_sensor_data():
    """
    Ingest raw sensor data from Unity Catalog Volume using Auto Loader.
    Auto Loader continuously monitors for new CSV files and processes them incrementally.
    """
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("header", "true")
            .option("cloudFiles.inferColumnTypes", "true")
            .load("/Volumes/workspace/default/smartcityinfradata/")
    )
