# Crypto Trading Bot

A production-ready cryptocurrency trading bot that implements a swing trading strategy for BTC/USD pairs using Coinbase's Advanced Trading API. The bot uses RSI and Moving Average crossover signals to identify trading opportunities.

## Features

- **Dual Mode Operation**
  - Simulation mode for testing strategies without real funds
  - Production mode for live trading
- **Trading Strategy**
  - RSI (Relative Strength Index) with configurable period
  - Moving Average crossover signals (short and long MA)
  - Customizable buy/sell thresholds
- **Infrastructure**
  - Docker-based local development
  - AWS Lambda deployment with EventBridge scheduling
  - Secure credentials management

## Project Structure
```
trading-bot/
├── src/
│   ├── bot_strategy/      # Trading strategy implementation
│   ├── coinbase_api/      # Coinbase API client
│   ├── aws_integration/   # AWS Lambda handler
│   └── local_run.py       # Local development entry point
├── infrastructure/        # Terraform AWS configuration
├── Dockerfile            # Local development container
├── docker-compose.yml    # Local development setup
└── requirements.txt      # Python dependencies
```

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Coinbase Advanced Trading API credentials
- AWS CLI (for deployment)
- Terraform (for deployment)

## Dependencies

Required Python packages (from requirements.txt):
pandas>=1.3.0
numpy>=1.21.0
requests>=2.26.0
boto3>=1.24.0
python-dotenv>=0.19.0


## Local Development Setup

1. **Clone the repository**
bash
git clone <repository-url>
cd trading-bot

2. **Configure API credentials**

Create a `secrets.json` file in the project root:
json:README.md
{
"api_key": "your_sandbox_api_key",
"api_secret": "your_sandbox_secret",
"passphrase": "your_sandbox_passphrase"
}

3. **Build and run with Docker**
```bash
# Build and start the bot
docker-compose up --build

# View logs
docker-compose logs -f
```

The bot will start in simulation mode, checking for trading signals every 5 minutes.

## Configuration

### Trading Parameters

Adjust strategy parameters in `src/bot_strategy/strategy.py`:
```python
class SwingStrategy:
    def __init__(self, 
                 rsi_period: int = 14,    # RSI calculation period
                 short_ma: int = 9,       # Short moving average period
                 long_ma: int = 21):      # Long moving average period
```

### Environment Variables

Configure in `docker-compose.yml`:
```yaml
environment:
  - TRADING_MODE=simulation  # 'simulation' or 'production'
  - SECRETS_FILE=/app/secrets.json
```

## AWS Deployment

### 1. Prepare Deployment Package
```bash
mkdir -p deployment
zip -r deployment/lambda.zip src/* requirements.txt
```

### 2. Configure AWS Credentials
```bash
aws configure
```

### 3. Store API Credentials
```bash
# For simulation mode
aws secretsmanager create-secret \
    --name trading-bot/simulation/coinbase-credentials \
    --secret-string '{"api_key":"sandbox_key","api_secret":"sandbox_secret","passphrase":"sandbox_passphrase"}'

# For production mode
aws secretsmanager create-secret \
    --name trading-bot/production/coinbase-credentials \
    --secret-string '{"api_key":"prod_key","api_secret":"prod_secret","passphrase":"prod_passphrase"}'
```

### 4. Deploy Infrastructure
```bash
cd infrastructure
terraform init
terraform plan
terraform apply
```

## Monitoring

### Local Development
```bash
# View real-time logs
docker-compose logs -f

# Check container status
docker-compose ps
```

### AWS Production
- CloudWatch Logs: `/aws/lambda/trading_bot`
- CloudWatch Metrics: Lambda execution metrics
- EventBridge: Execution schedule (every 4 hours)

## Troubleshooting

### Common Issues

1. **Docker Issues**
```bash
# Rebuild container
docker-compose build --no-cache
docker-compose up

# Check container logs
docker-compose logs -f
```

2. **AWS Deployment Issues**
- Check CloudWatch Logs for Lambda errors
- Verify IAM roles and permissions
- Confirm Secrets Manager values
- Check Lambda function timeout settings

3. **API Issues**
- Verify API credentials in secrets.json
- Check Coinbase API status
- Confirm network connectivity

## Security Best Practices

1. **API Credentials**
- Never commit secrets.json to version control
- Use AWS Secrets Manager for production
- Rotate API keys regularly

2. **AWS Security**
- Use least-privilege IAM roles
- Enable CloudWatch logging
- Encrypt sensitive data at rest

## Development Guidelines

1. **Testing Changes**
- Always test in simulation mode first
- Use sandbox API credentials for testing
- Monitor logs for unexpected behavior

2. **Making Changes**
- Update strategy parameters in strategy.py
- Modify risk management in lambda_handler.py
- Adjust scheduling in terraform configuration

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. Always test thoroughly in simulation mode before deploying with real funds.

## License

MIT License - See LICENSE file for details