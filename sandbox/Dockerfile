ARG BASE_IMAGE=python:3.12.14-slim
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY sandbox/requirements.txt /opt/sandbox-requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --timeout 120 --retries 5 --no-cache-dir -r /opt/sandbox-requirements.txt \
    && touch /opt/liangce-sandbox-worker
WORKDIR /app
COPY core /app/core
USER 65534:65534
ENTRYPOINT ["python", "-m", "core.sandbox_worker"]
