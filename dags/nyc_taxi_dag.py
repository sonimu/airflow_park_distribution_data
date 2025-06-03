from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import urllib.request

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet"
DEST = "/root/ACV_experiment/data/yellow_tripdata_2020-01.parquet"
DATA_CSV_DIR = "/root/ACV_experiment/data_csv"

SPARK_SCRIPT_PROCESAMIENTO = "/root/ACV_experiment/spark_scripts/procesamiento_spark.py"
SPARK_SCRIPT_REGRESION = "/root/ACV_experiment/spark_scripts/regresion_lineal.py"
SPARK_SCRIPT_KMEANS = "/root/ACV_experiment/spark_scripts/kmeans.py"
SPARK_SCRIPT_CONVERTIR_CSV = "/root/ACV_experiment/spark_scripts/parquet_to_csv.py"
SPARK_SCRIPT_GRAFICA_REGRESION = "/root/ACV_experiment/spark_scripts/load_and_plot_regresion.py"
SPARK_SCRIPT_GRAFICA_KMEANS = "/root/ACV_experiment/spark_scripts/load_and_plot_kmeans.py"

def descarga_parquet():
    if not os.path.exists(DEST):
        urllib.request.urlretrieve(URL, DEST)
        print(f"Descargado: {DEST}")
    else:
        print("Ya existe el archivo.")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='nyc_taxi_dag',
    default_args=default_args,
    schedule='@monthly',
    catchup=False,
    description="DAG para descargar y procesar datos NYC taxi, entrenar modelos y graficar resultados"
) as dag:

    descarga = PythonOperator(
        task_id='descarga_dataset',
        python_callable=descarga_parquet
    )

    procesa_spark = BashOperator(
        task_id='procesa_spark',
        bash_command=f'/opt/spark/bin/spark-submit {SPARK_SCRIPT_PROCESAMIENTO}'
    )
    #se ejecuta una sola vez, sino presenta problemas de sobrescritura de los datos
    #convertir_a_csv = BashOperator(
     #   task_id='convertir_parquet_a_csv',
      #  bash_command=f'/opt/spark/bin/spark-submit {SPARK_SCRIPT_CONVERTIR_CSV}'
    #)

    #entrena_regresion = BashOperator(
     #   task_id='entrena_regresion_csv',
      #  bash_command=f'/opt/spark/bin/spark-submit {SPARK_SCRIPT_REGRESION}'
    #)

    #entrena_kmeans = BashOperator(
     #   task_id='entrena_kmeans_csv',
      #  bash_command=f'/opt/spark/bin/spark-submit {SPARK_SCRIPT_KMEANS}'
    #)

    grafica_regresion = BashOperator(
        task_id='grafica_regresion',
        bash_command=f'/opt/spark/bin/spark-submit {SPARK_SCRIPT_GRAFICA_REGRESION}'
    )

    grafica_kmeans = BashOperator(
        task_id='grafica_kmeans',
        bash_command=f'/opt/spark/bin/spark-submit {SPARK_SCRIPT_GRAFICA_KMEANS}'
    )

    # Orden de ejecución
    descarga >> procesa_spark >> grafica_regresion >> grafica_kmeans





