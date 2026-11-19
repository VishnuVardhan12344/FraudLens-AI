
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def process_fraud_data(input_path: str):
    spark = (
        SparkSession.builder
        .appName("FraudLensAI")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        df = spark.read.parquet(input_path)

        print("Total transactions:", df.count())

        df.printSchema()

        fraud_summary = (
            df.groupBy("Class")
            .agg(
                F.count("*").alias("transaction_count"),
                F.avg("Amount").alias("average_amount"),
                F.sum("Amount").alias("total_amount"),
            )
            .orderBy("Class")
        )

        fraud_summary.show(truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "creditcard_features.parquet"
    )

    process_fraud_data(str(input_path))
