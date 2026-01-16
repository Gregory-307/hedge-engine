FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies for numpy/scipy wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first (leverages Docker layer cache)
COPY pyproject.toml /app/
RUN pip install --upgrade pip \
    && pip install .

# Copy application source code
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "hedge_engine.main:app", "--host", "0.0.0.0", "--port", "8000"] 