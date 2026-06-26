"""Kafka producer — streams CDR records onto the `cdr-stream` topic.

Simulates a live call feed by replaying the generated CDR dataset (looping),
so the Spark consumer / fraud detector has a continuous real-time source.

Run:  python -m pipeline.producer --rate 20
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from config.settings import configure_logging, get_settings

log = configure_logging("pipeline.producer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream CDRs to Kafka")
    parser.add_argument("--rate", type=float, default=20, help="records per second")
    args = parser.parse_args()

    settings = get_settings()
    from kafka import KafkaProducer  # imported lazily so the rest of the app needs no kafka

    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=5,
        linger_ms=50,
    )

    cdr_path = settings.data_dir / "cdrs.csv"
    if not cdr_path.exists():
        raise FileNotFoundError(f"{cdr_path} not found — run `python -m data.generate_data` first.")
    cdrs = pd.read_csv(cdr_path).to_dict("records")

    log.info("Streaming %d CDRs to topic '%s' at %.1f rec/s (Ctrl+C to stop)...",
             len(cdrs), settings.kafka_topic_cdr, args.rate)
    delay = 1.0 / max(args.rate, 0.1)
    sent = 0
    try:
        while True:  # loop the dataset forever to simulate a live feed
            for record in cdrs:
                producer.send(settings.kafka_topic_cdr, value=record)
                sent += 1
                if sent % 100 == 0:
                    log.info("Sent %d records", sent)
                time.sleep(delay)
    except KeyboardInterrupt:
        log.info("Stopping producer (sent %d records).", sent)
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
