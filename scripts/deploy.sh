#!/bin/bash

# Set up logging
exec 1> >(logger -s -t $(basename $0)) 2>&1

# Default values
MODE="local"
ENVIRONMENT="simulation"
VERBOSE=false
DETACH=false
REBUILD_BASE=false
TARGET="all"  # Default to building all containers
BUILD_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        --rebuild-base)
            REBUILD_BASE=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ ! "$MODE" =~ ^(local|aws)$ ]]; then
    echo "Invalid mode. Must be 'local' or 'aws'"
    exit 1
fi

if [[ ! "$ENVIRONMENT" =~ ^(simulation|production)$ ]]; then
    echo "Invalid environment. Must be 'simulation' or 'production'"
    exit 1
fi

if [[ ! "$TARGET" =~ ^(all|trading-bot|backtest)$ ]]; then
    echo "Invalid target. Must be 'all', 'trading-bot', or 'backtest'"
    exit 1
fi

# Execute Python deployment script
python3 "$(dirname "$0")/deploy.py" \
    --mode "$MODE" \
    --env "$ENVIRONMENT" \
    --target "$TARGET" \
    $([ "$VERBOSE" = true ] && echo "--verbose") \
    $([ "$DETACH" = true ] && echo "--detach") \
    $([ "$REBUILD_BASE" = true ] && echo "--rebuild-base") \
    $([ "$BUILD_ONLY" = true ] && echo "--build-only") 