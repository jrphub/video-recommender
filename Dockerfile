# Base image (compatible with implicit + scipy)
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (IMPORTANT for scipy/implicit)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY data/ data/
COPY vrmodels/ vrmodels/
COPY serving/ serving/

# Expose FastAPI port
EXPOSE 8000

# Start API
# CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8080
ENTRYPOINT ["python", "serve.py"]
