FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src/

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV TRADING_MODE=simulation
ENV PYTHONPATH=/app

# Command to run the bot
CMD ["python", "-m", "src.local_run"] 