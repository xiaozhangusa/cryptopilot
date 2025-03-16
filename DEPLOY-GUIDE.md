# How to Use Deployment Script with Backtest Container

The deployment script now supports building and running backtest containers. Here are some examples:

## Build and run backtest container
```bash
./scripts/deploy.sh --mode local --env simulation --target backtest
```

## Build only (don't run) the backtest container
```bash
./scripts/deploy.sh --mode local --env simulation --target backtest --build-only
```

## Force rebuild the base image and then the backtest container
```bash
./scripts/deploy.sh --mode local --env simulation --target backtest --rebuild-base
```

## Build all containers (trading-bot and backtest)
```bash
./scripts/deploy.sh --mode local --env simulation --target all --build-only
```

## Build and run in detached mode
```bash
./scripts/deploy.sh --mode local --env simulation --target backtest --detach
```

