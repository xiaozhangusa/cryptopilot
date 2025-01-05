variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "trading_mode" {
  description = "Trading mode (simulation or production)"
  type        = string
  default     = "simulation"
} 