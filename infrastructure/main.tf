terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Lambda function role
resource "aws_iam_role" "lambda_role" {
  name = "trading_bot_lambda_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Lambda function policy
resource "aws_iam_role_policy" "lambda_policy" {
  name = "trading_bot_lambda_policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [aws_secretsmanager_secret.coinbase_credentials_sim.arn,
                   aws_secretsmanager_secret.coinbase_credentials_prod.arn]
      }
    ]
  })
}

# Secrets for simulation mode
resource "aws_secretsmanager_secret" "coinbase_credentials_sim" {
  name = "trading-bot/simulation/coinbase-credentials"
}

# Secrets for production mode
resource "aws_secretsmanager_secret" "coinbase_credentials_prod" {
  name = "trading-bot/production/coinbase-credentials"
}

# Lambda function
resource "aws_lambda_function" "trading_bot" {
  filename         = "../deployment/lambda.zip"
  function_name    = "trading_bot"
  role            = aws_iam_role.lambda_role.arn
  handler         = "src.aws_integration.lambda_handler.lambda_handler"
  runtime         = "python3.9"
  timeout         = 300
  memory_size     = 256

  environment {
    variables = {
      TRADING_MODE = var.trading_mode
    }
  }
}

# EventBridge rule to trigger Lambda
resource "aws_cloudwatch_event_rule" "trading_schedule" {
  name                = "trading_bot_schedule"
  description         = "Schedule for trading bot execution"
  schedule_expression = "rate(4 hours)"
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.trading_schedule.name
  target_id = "TradingBotLambda"
  arn       = aws_lambda_function.trading_bot.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trading_bot.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.trading_schedule.arn
} 