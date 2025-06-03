from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

def main():
    spark = SparkSession.builder.appName("TaxiRegressionCSV").getOrCreate()

    # Leer el CSV
    df = spark.read.option("header", True).option("inferSchema", True).csv("/root/ACV_experiment/data_csv/part-00000-13d20ac9-4ae5-4f2e-b16b-a4d1e34d9082-c000.csv")

    # Mostrar el esquema (debug)
    df.printSchema()

    # Seleccionar columnas relevantes
    df = df.select("passenger_count", "trip_distance").dropna()

    # Asegurar que sean numéricos
    df = df.withColumn("passenger_count", col("passenger_count").cast("int")) \
           .withColumn("trip_distance", col("trip_distance").cast("double"))

    # Filtrar valores inválidos
    df = df.filter("trip_distance > 0 AND passenger_count > 0")

    print("Total de filas después del filtrado:", df.count())

    # Preparar features
    assembler = VectorAssembler(inputCols=["passenger_count"], outputCol="features")
    df = assembler.transform(df)

    # Split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(f"Filas de entrenamiento: {train_df.count()} - Test: {test_df.count()}")

    # Entrenar modelo
    lr = LinearRegression(featuresCol="features", labelCol="trip_distance")
    model = lr.fit(train_df)

    # Predicciones
    predictions = model.transform(test_df)

    # Evaluar
    evaluator = RegressionEvaluator(labelCol="trip_distance", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    print(f"RMSE en test set: {rmse}")

    # Guardar modelo (opcional)
    model.write().overwrite().save("/root/ACV_experiment/models/linear_regression_model")

    spark.stop()

if __name__ == "__main__":
    main()
