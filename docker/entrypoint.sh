#!/usr/bin/env bash
set -e

# The API trains data + models on first boot (idempotent). Other services
# (dashboard, producer, consumer) wait for the shared artifacts to appear so
# nothing races against an untrained model / missing data.

if [ "${TRAIN_ON_BOOT:-0}" = "1" ]; then
  if [ ! -f "${MODEL_DIR}/churn_model.joblib" ]; then
    echo "[entrypoint] Training models (first boot)..."
    python -m data.generate_data
    python -m models.train_all
  else
    echo "[entrypoint] Models already present — skipping training."
  fi
elif [ -n "${WAIT_FOR:-}" ]; then
  echo "[entrypoint] Waiting for ${WAIT_FOR} ..."
  until [ -f "${WAIT_FOR}" ]; do sleep 3; done
  echo "[entrypoint] Found ${WAIT_FOR} — continuing."
fi

exec "$@"
