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
   TRADING_MODE=simulation # Options: simulation, live
   TIMEFRAME=FIVE_MINUTE  # Options: FIVE_MINUTE, ONE_HOUR, SIX_HOUR, TWELVE_HOUR, ONE_DAY
   SECRETS_FILE=secrets.json
   ```

3. **Prepare Secrets**

   Create a `secrets.json` file with your Coinbase API credentials:

   ```json
   {
     "api_key": "your_api_key",
     "api_secret": "your_api_secret",
     "passphrase": "your_passphrase"
   }
   ```

## Building and Running

1. **First Time Setup** (or when dependencies change)

   ```bash
   ./scripts/deploy.sh --mode local --env simulation --rebuild-base
   ```

2. **Regular Development** (when only changing application code)

   ```bash
   # Run in foreground (logs show directly in terminal)
   ./scripts/deploy.sh --mode local --env simulation

   # Or run in background
   ./scripts/deploy.sh --mode local --env simulation --detach
   ```

3. **View Logs** (when running in detached mode)

   ```bash
   # View all logs
   docker-compose logs -f

   # View last 100 lines
   docker-compose logs -f --tail=100
   ```

   Press Ctrl+C to stop viewing logs (container will continue running in detached mode)

## Development

The base image contains all dependencies and system setup. You only need to rebuild it when:
- Setting up the project for the first time
- Updating dependencies in requirements.txt or requirements-dev.txt
- Making changes to Dockerfile.base

For regular development where you're only changing application code, you don't need the --rebuild-base flag.

## Available Deploy Script Options

- `--mode`: Choose deployment mode (`local` or `aws`)
- `--env`: Choose environment (`simulation` or `live`)
- `--verbose`: Enable verbose logging
- `--detach`: Run containers in background
- `--rebuild-base`: Force rebuild of the base image

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

- `TRADING_MODE`: Set to 'simulation' or 'live' (default: simulation)
- `SECRETS_FILE`: Path to secrets file (default: secrets.json)
- `PYTHONPATH`: Python path for module imports

## Trading Strategy

The bot implements a swing trading strategy based on:
- RSI (Relative Strength Index)
- Support and Resistance levels
- Risk/Reward analysis

### Trading Criteria

A trade signal is only executed when ALL of the following criteria are met:

1. **Technical Conditions**
   - For BUY signals:
     - Price near identified support level
     - RSI indicates oversold condition
     - Clear potential upside identified
   - For SELL signals:
     - Price near identified resistance level
     - RSI indicates overbought condition
     - Clear potential downside identified

2. **Risk Management**
   - Risk/Reward ratio ≥ 1:2 (minimum)
   - Profit/Cost ratio > 5%
   - Net profit positive after fees
   - Stop loss within timeframe limits:
     - 5-minute: 1%
     - 1-hour: 2%
     - 6-hour: 3%
     - 12-hour: 4%
     - 1-day: 5%

3. **Risk Rating System**
   - 🌟 Excellent: 1:3+ ratio, >10% profit potential
   - ✅ Good: 1:2+ ratio, >5% profit potential
   - ⚠️ Marginal: 1:1.5+ ratio, >2% profit potential
   - ❌ Poor: Anything less

The strategy adapts its parameters based on the selected timeframe to optimize for different trading horizons while maintaining strict risk management principles.

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
./scripts/deploy.sh --mode aws --env live
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