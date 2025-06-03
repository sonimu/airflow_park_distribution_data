from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col

def main():
    spark = SparkSession.builder.appName("TaxiKMeansClustering").getOrCreate()

    # Leer CSV
    df = spark.read.option("header", True).option("inferSchema", True).csv("/root/ACV_experiment/data_csv/part-00000-13d20ac9-4ae5-4f2e-b16b-a4d1e34d9082-c000.csv")

    # Preprocesamiento
    df = df.select("passenger_count", "trip_distance").dropna()
    df = df.withColumn("passenger_count", col("passenger_count").cast("int")) \
           .withColumn("trip_distance", col("trip_distance").cast("double"))
    df = df.filter("trip_distance > 0 AND passenger_count > 0")

    print("Total de filas para clustering:", df.count())

    # Armar vector de características
    assembler = VectorAssembler(inputCols=["passenger_count", "trip_distance"], outputCol="features")
    df_features = assembler.transform(df)

    # KMeans
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=4, seed=42)
    model = kmeans.fit(df_features)

    # Resultados
    clustered = model.transform(df_features)
    clustered.select("passenger_count", "trip_distance", "cluster").show(10)

    # Guardar modelo
    model.write().overwrite().save("/root/ACV_experiment/models/kmeans_model")

    spark.stop()

if __name__ == "__main__":
    main()
