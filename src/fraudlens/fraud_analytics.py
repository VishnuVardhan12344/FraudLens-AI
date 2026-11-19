
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def analyze_fraud(input_path: str):

    spark = (
        SparkSession.builder
        .appName("FraudLensAdvancedAnalytics")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        df = spark.read.parquet(input_path)

        # Cache because multiple actions reuse the dataset.
        df = df.cache()

        total_transactions = df.count()

        if total_transactions == 0:
            raise ValueError("Dataset is empty")

        # Fraud transactions.
        fraud_df = df.filter(F.col("Class") == 1)

        fraud_count = fraud_df.count()

        fraud_rate = (
            fraud_count / total_transactions
        ) * 100

        print("\n===== FRAUD KPIs =====")
        print("Total transactions:", total_transactions)
        print("Fraud transactions:", fraud_count)
        print(f"Fraud rate: {fraud_rate:.4f}%")

        # High-value fraud transactions.
        high_value_fraud = fraud_df.filter(
            F.col("Amount") > 500
        )

        print("\nFraud transactions above $500:")
        print(high_value_fraud.count())

        # Compare transaction amounts by class.
        amount_summary = (
            df.groupBy("Class")
            .agg(
                F.count("*").alias("transaction_count"),
                F.avg("Amount").alias("average_amount"),
                F.sum("Amount").alias("total_amount"),
                F.max("Amount").alias("maximum_amount")
            )
            .orderBy("Class")
        )

        print("\n===== AMOUNT ANALYSIS =====")
        amount_summary.show(truncate=False)

        # Fraud trends by elapsed day.
        fraud_by_day = (
            df.groupBy("elapsed_day")
            .agg(
                F.count("*").alias("total_transactions"),
                F.sum(
                    F.when(F.col("Class") == 1, 1)
                    .otherwise(0)
                ).alias("fraud_transactions")
            )
            .withColumn(
                "fraud_rate_pct",
                F.round(
                    F.col("fraud_transactions")
                    / F.col("total_transactions") * 100,
                    4
                )
            )
            .orderBy("elapsed_day")
        )

        print("\n===== FRAUD BY ELAPSED DAY =====")
        fraud_by_day.show(truncate=False)

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

    analyze_fraud(str(input_path))
