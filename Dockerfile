FROM python:3.9-slim

WORKDIR /app

# Install gcc for any compiled dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project first
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Install the package in development mode
RUN pip install -e .[dev]

# Environment variables
ENV TRADING_MODE=simulation
ENV PYTHONPATH=/app/src

# Command to run the bot
CMD ["python", "-m", "local_run"] 