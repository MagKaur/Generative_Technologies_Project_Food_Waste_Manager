FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj kod źródłowy (src/ do /app/src/)
COPY src/ /app/src/


EXPOSE 8080
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]