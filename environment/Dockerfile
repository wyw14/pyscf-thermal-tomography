FROM --platform=linux/amd64 python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
WORKDIR /app
COPY wheels/ /opt/runtime-wheels/
COPY requirements.lock.txt /tmp/runtime-requirements.lock.txt
RUN python -m pip install --no-cache-dir --no-index --find-links=/opt/runtime-wheels --require-hashes -r /tmp/runtime-requirements.lock.txt && python -m pip check
# Runtime-only filesystem: hidden tests and solutions never enter it.
RUN mkdir -p /opt/candidate-root/etc /opt/candidate-root/tmp \
        /opt/candidate-root/dev /opt/candidate-root/app \
    && cp -al /usr /opt/candidate-root/usr \
    && cp -a /bin /lib /lib64 /opt/candidate-root/ \
    && cp /etc/ld.so.cache /opt/candidate-root/etc/ld.so.cache \
    && chmod 0755 /opt/candidate-root /opt/candidate-root/tmp
COPY src/ /app/src/
COPY runner/ /runner/
COPY runner/ /opt/candidate-root/runner/
ENV PYTHONPATH=/app/src
