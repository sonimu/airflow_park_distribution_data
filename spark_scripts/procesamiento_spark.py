from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TaxiParquetProcessing") \
    .getOrCreate()

df = spark.read.parquet("/root/ACV_experiment/data/yellow_tripdata_2020-01.parquet")
df = df.dropna()
df.select("trip_distance", "passenger_count").show(10)

df.write.mode("overwrite").parquet("/root/ACV_experiment/data/processed_taxi_data")
