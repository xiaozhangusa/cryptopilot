# CryptoPilot Trading Bot

A cryptocurrency trading bot that implements a swing trading strategy using Coinbase's Advanced Trading API.

## Features

- **Swing Trading Strategy**: Utilizes the Relative Strength Index (RSI) to identify overbought and oversold conditions.
- **Multiple Timeframes**: Supports flexible trading strategies with configurable timeframes, including 5-minute, 1-hour, 6-hour, 12-hour, and 1-day intervals.
- **Simulation Mode**: Test strategies without executing real trades.
- **Docker Support**: Easily deployable using Docker and Docker Compose.

## Project Structure

```
cryptopilot/
├── src/                      # Source code
│   ├── bot_strategy/        # Trading strategy implementation
│   ├── coinbase_api/        # Coinbase API client
│   ├── aws_integration/     # AWS Lambda integration
│   └── local_run.py         # Local execution script
├── tests/                   # Test files
├── infrastructure/          # Terraform AWS infrastructure
├── scripts/                 # Deployment and utility scripts
├── docs/                    # Documentation
└── docker/                  # Docker configuration files
```

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ installed
- Coinbase API credentials

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo/trading-bot.git
   cd trading-bot
   ```

2. **Configure Environment Variables**

   Create a `.env` file in the root directory with the following content:

   ```env
   TRADING_MODE=simulation
   TIMEFRAME=FIVE_MINUTE  # Options: FIVE_MINUTE, ONE_HOUR, SIX_HOUR, TWELVE_HOUR, ONE_DAY
   SECRETS_FILE=secrets.json
   ```

3. **Prepare Secrets**

   Ensure you have a `secrets.json` file with your Coinbase API credentials:

   ```json
   {
     "api_key": "your_api_key",
     "api_secret": "your_api_secret",
     "passphrase": "your_passphrase"
   }
   ```

## Running the Bot

### Using Docker Compose

1. **Build and Start the Container**

   ```bash
   docker-compose up --build
   ```

   This will start the trading bot in the specified timeframe and mode.

2. **Logs and Monitoring**

   You can view logs in the terminal to monitor the bot's activity:

   ```bash
   docker logs trading-bot-1 -f
   ```

### Running Locally

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Bot**

   ```bash
   python -m src.local_run
   ```

   Ensure the environment variables are set in your terminal session or `.env` file.

## Strategy Overview

The bot uses a Swing Trading Strategy with RSI (Relative Strength Index) tailored for different timeframes:

- **5-Minute**: More sensitive, uses RSI thresholds of 25/75.
- **1-Hour**: Standard thresholds of 30/70.
- **6-Hour**: Less sensitive, uses thresholds of 35/65.
- **12-Hour**: Similar to 6-hour, with thresholds of 35/65.
- **1-Day**: Most conservative, uses thresholds of 40/60.

## Customization

- **Timeframes**: Adjust the `TIMEFRAME` environment variable to switch between different trading timeframes.
- **RSI Period**: Modify the `rsi_period` in `src/bot_strategy/strategy.py` if needed.
- **Stop Loss**: Adjust stop loss percentages in `src/bot_strategy/trade_analyzer.py` based on your risk tolerance.

## Troubleshooting

- Ensure Docker and Python are correctly installed and configured.
- Verify API credentials in `secrets.json`.
- Check logs for any error messages and adjust configurations accordingly.

For further assistance, refer to the project's issue tracker or contact the development team.

## Configuration

The bot can be configured using environment variables:

- `TRADING_MODE`: Set to 'simulation' or 'production' (default: simulation)
- `SECRETS_FILE`: Path to secrets file (default: secrets.json)
- `PYTHONPATH`: Python path for module imports

## Trading Strategy

The bot implements a swing trading strategy based on:
- RSI (Relative Strength Index)
- Moving Average Crossover
- Price action analysis

Configure strategy parameters in `src/bot_strategy/strategy.py`.

## API Integration

The bot uses Coinbase's Advanced Trading API v3. Key features:
- Real-time market data
- Order management
- Account information
- Historical price data

## Testing

Run tests using:
```bash
docker-compose --profile test up
```

## Deployment

### Local Deployment
```bash
./scripts/deploy.sh --mode local --env simulation
```

### AWS Deployment
```bash
# Configure AWS credentials first
./scripts/deploy.sh --mode aws --env production
```

## Monitoring

- Local logs available via `docker-compose logs -f`
- AWS CloudWatch logs for cloud deployment
- Trading performance metrics in CloudWatch

## Security

- API credentials are stored securely in secrets.json (local) or AWS Secrets Manager (cloud)
- All API requests use HMAC SHA-256 signatures
- Environment-specific configurations for isolation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. Cryptocurrency trading involves substantial risk of loss.