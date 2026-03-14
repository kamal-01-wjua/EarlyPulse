FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY api/       ./api/
COPY src/       ./src/
COPY config/    ./config/
COPY *.json     ./  2>/dev/null || true
COPY *.npy      ./  2>/dev/null || true
COPY data/      ./data/  2>/dev/null || true
COPY experiments/ ./experiments/  2>/dev/null || true

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
