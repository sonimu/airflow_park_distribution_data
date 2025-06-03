from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import os

def main():
    spark = SparkSession.builder.appName("LoadAndPlotKMeans").getOrCreate()

    # Cargo el dataset CSV que usaste para KMeans
    df = spark.read.csv("/root/ACV_experiment/data_csv", header=True, inferSchema=True)

    from pyspark.ml.clustering import KMeansModel
    from pyspark.ml.feature import VectorAssembler

    feature_cols = ["passenger_count", "trip_distance"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_feat = assembler.transform(df).na.drop()

    model_path = "/root/ACV_experiment/models/kmeans_model"
    kmeans_model = KMeansModel.load(model_path)

    predictions = kmeans_model.transform(df_feat)

    pandas_df = predictions.select("passenger_count", "trip_distance", "prediction").toPandas()

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pandas_df["passenger_count"], pandas_df["trip_distance"], c=pandas_df["prediction"], cmap="viridis")
    plt.xlabel("Passenger Count")
    plt.ylabel("Trip Distance")
    plt.title("Clusters KMeans")
    plt.colorbar(scatter, label="Cluster")
    
    output_dir = "/root/ACV_experiment/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "kmeans_model_loaded.png"))
    plt.close()

    spark.stop()

if __name__ == "__main__":
    main()
