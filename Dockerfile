# Use the locally built base image
FROM localhost/trading-bot-base:latest as final

WORKDIR /app

# Copy the entire project
COPY . /app

# Install the package in development mode
RUN pip install -e .[dev]

# Environment variables
ENV TRADING_MODE=simulation
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Command to run the bot
CMD ["python", "-u", "-m", "src.local_run"]

# Make port 80 available to the world outside this container
EXPOSE 80 