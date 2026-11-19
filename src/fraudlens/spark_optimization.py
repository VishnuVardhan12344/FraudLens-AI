
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def analyze_spark_performance(input_path: str):

    spark = (
        SparkSession.builder
        .appName("FraudLensSparkOptimization")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    try:
        df = spark.read.parquet(input_path)

        print("\nInitial partitions:")
        print(df.rdd.getNumPartitions())

        # Transformations are lazy.
        filtered_df = df.filter(
            F.col("Amount") > 100
        )

        selected_df = filtered_df.select(
            "Time",
            "Amount",
            "Class",
            "log_amount"
        )

        # Inspect Spark's execution plan.
        print("\nSpark execution plan:")
        selected_df.explain(mode="formatted")

        # Action: executes the computation.
        print("\nTransactions above $100:")
        print(selected_df.count())

        # Repartition data for parallel processing.
        repartitioned_df = df.repartition(8)

        print("\nPartitions after repartition:")
        print(repartitioned_df.rdd.getNumPartitions())

        # Fraud statistics.
        fraud_stats = (
            repartitioned_df
            .groupBy("Class")
            .agg(
                F.count("*").alias("total_transactions"),
                F.avg("Amount").alias("average_amount"),
                F.max("Amount").alias("maximum_amount")
            )
        )

        fraud_stats.show(truncate=False)

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

    analyze_spark_performance(str(input_path))
