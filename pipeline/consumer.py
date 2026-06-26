"""Lightweight Kafka consumer (no Spark) — real-time fraud scoring.

A drop-in alternative to `spark_consumer.py` for machines without Spark.
Consumes the `cdr-stream` topic, scores each CDR with the fraud model, and logs
alerts in under 2 seconds per record.

Run:  python -m pipeline.consumer
"""
from __future__ import annotations

import json
import time

from config.settings import configure_logging, get_settings

log = configure_logging("pipeline.consumer")


def main() -> None:
    settings = get_settings()
    from kafka import KafkaConsumer

    from models.fraud import FraudDetector

    detector = FraudDetector()
    consumer = KafkaConsumer(
        settings.kafka_topic_cdr,
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="telecomiq-fraud",
    )
    log.info("Consuming CDRs from '%s' for real-time fraud scoring...", settings.kafka_topic_cdr)

    processed = alerts = 0
    for message in consumer:
        start = time.perf_counter()
        verdict = detector.score_one(message.value)
        latency_ms = (time.perf_counter() - start) * 1000
        processed += 1
        if verdict.is_alert:
            alerts += 1
            log.warning(
                "🚨 [%s] %s score=%.2f (%.1fms) — %s",
                verdict.severity, verdict.cdr_id, verdict.anomaly_score,
                latency_ms, "; ".join(verdict.reasons),
            )
        if processed % 200 == 0:
            log.info("Processed %d CDRs | %d alerts", processed, alerts)


if __name__ == "__main__":
    main()
