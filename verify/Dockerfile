# Verification image: Python + git + common build backends.
# The competition sandbox uses Python 3.13; we default to 3.12 for older commits
# and re-verify final tasks inside the official sandbox image once data is downloaded.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest setuptools wheel hatchling flit-core poetry-core pdm-backend
