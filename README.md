# CryptoPilot Trading Bot

A cryptocurrency trading bot that implements a swing trading strategy using Coinbase's Advanced Trading API.

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

- Python 3.9+
- Docker
- Coinbase Advanced Trading API credentials
- AWS account (for cloud deployment)

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cryptopilot.git
cd cryptopilot
```

2. Create a `secrets.json` file in the project root:
```json
{
    "api_key": "your_coinbase_api_key",
    "api_secret": "your_coinbase_api_secret",
    "passphrase": "your_coinbase_passphrase"
}
```

3. Build and run using Docker:
```bash
docker-compose build
docker-compose up
```

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