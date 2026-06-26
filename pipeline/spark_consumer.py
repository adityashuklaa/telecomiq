"""Spark Structured Streaming consumer — real-time fraud scoring on the CDR feed.

Reads the `cdr-stream` Kafka topic, parses each CDR, scores it with the trained
fraud model, and writes ALERT rows to the console sink (swap for Delta Lake /
PostgreSQL in production — see the commented sink at the bottom).

Run (with the Kafka package):
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2 \
      pipeline/spark_consumer.py

A lightweight, Spark-free fallback consumer is in `pipeline/consumer.py` for
machines without a Spark install.
"""
from __future__ import annotations

from config.settings import configure_logging, get_settings

log = configure_logging("pipeline.spark")


def main() -> None:
    settings = get_settings()
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    from models.fraud import FraudDetector

    detector = FraudDetector()  # loaded on the driver; broadcast via closure

    spark = (
        SparkSession.builder.appName("TelecomIQ-FraudStream")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType(
        [
            StructField("cdr_id", StringType()),
            StructField("caller_id", StringType()),
            StructField("hour_of_day", IntegerType()),
            StructField("duration_sec", DoubleType()),
            StructField("cost", DoubleType()),
            StructField("is_international", IntegerType()),
            StructField("is_premium_rate", IntegerType()),
            StructField("calls_last_hour", IntegerType()),
            StructField("unique_numbers_called", IntegerType()),
        ]
    )

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.kafka_topic_cdr)
        .option("startingOffsets", "latest")
        .load()
    )
    parsed = raw.select(F.from_json(F.col("value").cast("string"), schema).alias("d")).select("d.*")

    def score_partition(rows):
        """Score each CDR in the partition; yield only fraud alerts."""
        for row in rows:
            verdict = detector.score_one(row.asDict())
            if verdict.is_alert:
                yield (verdict.cdr_id, verdict.anomaly_score, verdict.severity,
                       "; ".join(verdict.reasons))

    def process_batch(batch_df, batch_id: int):
        alerts = batch_df.rdd.mapPartitions(score_partition).collect()
        if alerts:
            log.warning("Batch %d — %d FRAUD ALERTS", batch_id, len(alerts))
            for cdr_id, score, severity, reasons in alerts[:10]:
                log.warning("  [%s] %s score=%.2f — %s", severity, cdr_id, score, reasons)

    query = (
        parsed.writeStream.foreachBatch(process_batch)
        .outputMode("append")
        .option("checkpointLocation", "/tmp/telecomiq-fraud-ckpt")
        .start()
    )
    log.info("Spark fraud stream started. Awaiting CDRs on '%s'...", settings.kafka_topic_cdr)
    query.awaitTermination()

    # ── Production sink example (Delta Lake) ─────────────────────────────
    # parsed.writeStream.format("delta") \
    #   .option("checkpointLocation", "/data/_ckpt/cdr") \
    #   .start("/data/delta/cdr_alerts")


if __name__ == "__main__":
    main()
