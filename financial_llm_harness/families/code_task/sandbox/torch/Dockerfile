# Sandbox image "torch" (spec §13.4, E-22, §T6.2 K.M.3): the base packages plus CPU PyTorch.
# The base digest is python:3.12-slim as pulled on 2026-10-07 (multi-platform index digest).
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
