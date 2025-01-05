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
- Docker Compose V2 (comes with Docker Desktop)
- Python 3.9+
- Coinbase Advanced Trading API credentials
- AWS CLI (for deployment)
- Terraform (for deployment)

## Dependencies

Required Python packages (from requirements.txt):
```
pandas>=1.3.0
numpy>=1.21.0
requests>=2.26.0
boto3>=1.24.0
python-dotenv>=0.19.0
```

## Local Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd trading-bot
```

2. **Configure API credentials**

Create a `secrets.json` file in the project root:
```json
{
"api_key": "your_sandbox_api_key",
"api_secret": "your_sandbox_secret",
"passphrase": "your_sandbox_passphrase"
}
```

3. **Build and run with Docker**
```bash
# Build and start the bot
docker-compose up --build

# View logs
docker-compose logs -f
```

4. **Running Tests**
```bash
# Run all tests
docker-compose run test

# Run specific test file
docker-compose run test python -m pytest tests/test_specific.py

# Run tests with coverage
docker-compose run test python -m pytest --cov=src tests/
```

The test environment:
- Uses a separate Dockerfile (`Dockerfile.test`)
- Includes additional test dependencies
- Mounts test directory for real-time test development

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
services:
  trading-bot:
    environment:
      - TRADING_MODE=simulation  # 'simulation' or 'production'
      - SECRETS_FILE=/app/secrets.json
```

### Test Configuration

Test dependencies are managed in `requirements-dev.txt`:
```text
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
```

Test environment variables can be configured in `docker-compose.yml`:
```yaml
services:
  test:
    build:
      context: .
      dockerfile: Dockerfile.test
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    command: python -m pytest
```

## AWS Deployment

## Deployment

The project includes a deployment script that handles both local and AWS deployments. The script provides detailed logging and error handling.

### Prerequisites

For local deployment:
- Docker and Docker Compose
- Python 3.9+
- Coinbase API credentials in `secrets.json`

For AWS deployment:
- AWS CLI configured
- Terraform installed
- AWS credentials with appropriate permissions
- Secrets configured in AWS Secrets Manager

### Usage

```bash
# Make the deployment scripts executable
chmod +x scripts/deploy.sh
chmod +x scripts/deploy.py

# Local deployment in simulation mode
# Interactive mode (see logs in real-time)
./scripts/deploy.sh --mode local --env simulation

# Detached mode (run in background)
./scripts/deploy.sh --mode local --env simulation --detach

# Local deployment in production mode
./scripts/deploy.sh --mode local --env production [--detach]

# AWS deployment in simulation mode
./scripts/deploy.sh --mode aws --env simulation [--detach]

# AWS deployment in production mode
./scripts/deploy.sh --mode aws --env production [--detach]

# Enable verbose logging with -v flag
./scripts/deploy.sh --mode aws --env simulation -v [--detach]
```

### Deployment Options

- `--mode`: Specify deployment target (`local` or `aws`)
- `--env`: Specify environment (`simulation` or `production`)
- `-v, --verbose`: Enable detailed logging output
- `-d, --detach`: Run containers in background mode

### Deployment Process

The deployment script will:

1. **Check Prerequisites**
   - Verify required tools are installed
   - Check AWS credentials (for AWS deployment)
   - Validate Docker installation (for local deployment)

2. **Validate Secrets**
   - Check presence and format of secrets.json (local)
   - Verify AWS Secrets Manager configuration (AWS)

3. **Deploy Application**
   - Local: Build and run Docker containers (interactive or detached mode)
   - AWS: Create deployment package and apply Terraform configuration

4. **Verify Deployment**
   - Local: Confirm containers are running
   - AWS: Verify Lambda function deployment

### Monitoring Deployment

Local deployment:
```bash
# If running in interactive mode:
# Logs will appear automatically in the console

# If running in detached mode:
# View container logs
docker-compose logs -f

# Check container status
docker-compose ps

# Stop containers
docker-compose down
```

AWS deployment:
- Check CloudWatch Logs in AWS Console
- Monitor Lambda function metrics
- View EventBridge execution history

### Troubleshooting

The deployment script creates a `deployment.log` file with detailed information about the deployment process.

Common issues:

1. **Local Deployment**
   ```bash
   # Rebuild containers
   docker-compose build --no-cache
   
   # Restart containers
   docker-compose restart
   ```

2. **AWS Deployment**
   ```bash
   # Clean up Terraform state
   cd infrastructure
   terraform destroy
   terraform init -reconfigure
   
   # Redeploy
   ./scripts/deploy.sh --mode aws --env simulation
   ```

### Cleanup

Local environment:
```bash
docker-compose down
```

AWS environment:
```bash
cd infrastructure
terraform destroy
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