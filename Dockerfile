FROM python:3.12-slim

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /app

# Deps najpierw (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiuj TYLKO src/ (dzięki .dockerignore – reszta ignorowana)
COPY src/ /app/src/

# Domyślny entrypoint
CMD ["python", "src/main.py"]