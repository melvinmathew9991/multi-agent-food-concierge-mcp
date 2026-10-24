# syntax=docker/dockerfile:1
# The HTTP server (API + MCP) for a Hugging Face Space (Docker SDK, free CPU) or any container host.
#
#   docker build -t food-concierge .
#   docker run --rm -p 7860:7860 --env-file .env food-concierge     # keys from .env, never baked in
#   python scripts/smoke_deploy.py http://127.0.0.1:7860
#
# The catalog, indexes, embedding model and photo thumbnails are built into the image, so a cold start reads only
# local files and needs no network except to the model providers. The base image is pinned by digest; packages
# come from the hash-checked locks.

ARG PYTHON_IMAGE=python:3.11-slim-trixie@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377

# --- build: the offline pipeline needs the ingest extra (pandas, prefect), which the server never imports ---------
FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src MODEL_CACHE_DIR=/app/cache/models PHOTO_CACHE_DIR=/app/cache/photos \
    PREFECT_HOME=/tmp/prefect
WORKDIR /app
COPY requirements-dev.lock ./
# --no-deps: the lock pins every package, and pip would reject the uv Pillow override (see CHANGELOG, #98).
RUN python -m venv /opt/build \
    && /opt/build/bin/pip install --require-hashes --no-deps -r requirements-dev.lock
COPY pyproject.toml ./
COPY src/ src/
COPY data/ data/
COPY scripts/fetch_photos.py scripts/
# Catalog and indexes (downloads the embedding model), then the photos, verified against their pinned SHA-256.
# Only the thumbnails are kept: the originals are large and nothing at runtime reads them.
RUN /opt/build/bin/python -m food_concierge.flows build \
    && /opt/build/bin/python scripts/fetch_photos.py sync \
    && rm -rf /app/cache/photos/originals

# --- runtime: the server's own lock, a non-root user, only what it reads ------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src \
    MODEL_CACHE_DIR=/app/cache/models PHOTO_CACHE_DIR=/app/cache/photos \
    HF_HUB_OFFLINE=1 \
    ENVIRONMENT=production API_HOST=0.0.0.0 API_PORT=7860
COPY requirements.lock /tmp/
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes --no-deps -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app/src src/
COPY --from=build --chown=app:app /app/data data/
COPY --from=build --chown=app:app /app/cache cache/
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"]
CMD ["python", "-m", "food_concierge.api"]
