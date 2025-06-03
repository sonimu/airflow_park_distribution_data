from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel

def main():
    spark = SparkSession.builder.appName("LoadAndPlotRegression").getOrCreate()

    # Carga el CSV, igual que en entrenamiento
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        "/root/ACV_experiment/data_csv/part-00000-13d20ac9-4ae5-4f2e-b16b-a4d1e34d9082-c000.csv"
    )

    feature_cols = ["passenger_count"]
    label_col = "trip_distance"

    df_clean = df.select(feature_cols + [label_col]).na.drop()

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_feat = assembler.transform(df_clean)

    # Carga el modelo entrenado
    model_path = "/root/ACV_experiment/models/linear_regression_model"
    lr_model = LinearRegressionModel.load(model_path)

    predictions = lr_model.transform(df_feat)

    pandas_df = predictions.select("passenger_count", label_col, "prediction").toPandas()

    plt.scatter(pandas_df["passenger_count"], pandas_df[label_col], color="blue", label="Actual")
    plt.scatter(pandas_df["passenger_count"], pandas_df["prediction"], color="red", label="Predicción", alpha=0.5)
    plt.xlabel("Passenger Count")
    plt.ylabel("Trip Distance")
    plt.title("Regresión Lineal: Trip Distance vs Passenger Count")
    plt.legend()

    output_dir = "/root/ACV_experiment/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "regresion_model_loaded.png"))
    plt.close()

    spark.stop()

if __name__ == "__main__":
    main()
