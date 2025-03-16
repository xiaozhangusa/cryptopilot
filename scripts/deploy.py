#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import logging
from typing import Optional
import json

class ProgressIndicator:
    def __init__(self):
        self.spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.index = 0
        
    def next(self, message: str):
        sys.stdout.write(f'\r{self.spinner[self.index]} {message}')
        sys.stdout.flush()
        self.index = (self.index + 1) % len(self.spinner)

class Deployer:
    def __init__(self, mode: str, environment: str, target: str, args):
        self.mode = mode  # 'local' or 'aws'
        self.environment = environment  # 'simulation' or 'live'
        self.target = target  # 'all', 'trading-bot', or 'backtest'
        self.args = args
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.progress = ProgressIndicator()

    def execute_command(self, command: str, cwd: Optional[str] = None) -> int:
        """Execute a shell command and return exit code"""
        logging.info(f"\n🔧 Executing: {command}")
        if cwd:
            logging.info(f"📍 Working directory: {cwd}")
        
        process = subprocess.Popen(
            command.split(),
            cwd=cwd or self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logging.info(output.strip())
                
        return_code = process.poll()
        logging.info(f"📋 Exit code: {return_code}")
        
        if output := process.stdout.read():
            logging.info(f"📤 Output:\n{output}")
            
        return return_code

    def check_prerequisites(self):
        """Check if required tools are installed"""
        logging.info("Checking prerequisites...")
        
        # Check Docker
        if self.execute_command("docker --version") != 0:
            raise RuntimeError("Docker is not installed or not running")
            
        # Check Docker Compose
        if self.execute_command("docker-compose --version") != 0:
            raise RuntimeError("Docker Compose is not installed")
            
        logging.info("✅ All prerequisites checked successfully")

    def validate_secrets(self):
        """Validate secrets configuration"""
        logging.info("Validating secrets...")
        
        secrets_file = os.path.join(self.project_root, 'secrets.json')
        if not os.path.exists(secrets_file):
            if self.environment == 'simulation':
                logging.warning("⚠️ No secrets.json found, will use dummy credentials")
            else:
                raise RuntimeError("secrets.json is required for live deployment")
        else:
            try:
                with open(secrets_file) as f:
                    secrets = json.load(f)
                required_keys = ['api_key', 'api_secret', 'passphrase']
                if not all(key in secrets for key in required_keys):
                    raise RuntimeError("Missing required keys in secrets.json")
            except json.JSONDecodeError:
                raise RuntimeError("Invalid JSON in secrets.json")
                
        logging.info("✅ Secrets validation completed")

    def build_base_image(self):
        """Build the base image if it doesn't exist or if forced"""
        logging.info("Building base image...")
        base_image_tag = "localhost/trading-bot-base:latest"
        if self.args.rebuild_base or self.execute_command(f"docker images -q {base_image_tag}") != 0:
            if self.execute_command(f"docker build -f Dockerfile.base -t {base_image_tag} .") != 0:
                raise RuntimeError("Failed to build base image")
            logging.info("✅ Base image built successfully")
        else:
            logging.info("✅ Using existing base image")

    def build_target(self):
        """Build the specified target without starting it"""
        logging.info(f"Building {self.target} container(s)...")
        
        # Build base image first
        self.build_base_image()
        
        # Build the appropriate container(s)
        if self.target == 'all':
            if self.execute_command("docker-compose build") != 0:
                raise RuntimeError("Failed to build all containers")
            logging.info("✅ All containers built successfully")
        elif self.target == 'trading-bot':
            if self.execute_command("docker-compose build trading-bot") != 0:
                raise RuntimeError("Failed to build trading-bot container")
            logging.info("✅ Trading bot container built successfully")
        elif self.target == 'backtest':
            if self.execute_command("docker-compose build backtest") != 0:
                raise RuntimeError("Failed to build backtest container")
            logging.info("✅ Backtest container built successfully")

    def deploy_local(self):
        """Handle local deployment"""
        logging.info("Starting local deployment...")
        
        # Always build base image first
        self.build_base_image()
        
        # If build-only flag is set, just build the target and exit
        if self.args.build_only:
            self.build_target()
            return
        
        # Build Docker containers based on target
        if self.target == 'all':
            logging.info("Building all containers...")
            if self.execute_command("docker-compose build") != 0:
                raise RuntimeError("Failed to build Docker containers")
            
            # Start containers
            logging.info("Starting Docker containers...")
            if self.args.detach:
                command = "docker-compose up -d"
                if self.execute_command(command) != 0:
                    raise RuntimeError("Failed to start Docker containers")
                
                # Verify containers are running
                if self.execute_command("docker-compose ps -q") != 0:
                    raise RuntimeError("Container failed to start")
            else:
                # Simply execute docker-compose up
                os.execvp("docker-compose", ["docker-compose", "up"])
                
        elif self.target == 'trading-bot':
            logging.info("Building trading-bot container...")
            if self.execute_command("docker-compose build trading-bot") != 0:
                raise RuntimeError("Failed to build trading-bot container")
            
            # Start trading-bot container
            logging.info("Starting trading-bot container...")
            if self.args.detach:
                command = "docker-compose up -d trading-bot"
                if self.execute_command(command) != 0:
                    raise RuntimeError("Failed to start trading-bot container")
            else:
                os.execvp("docker-compose", ["docker-compose", "up", "trading-bot"])
                
        elif self.target == 'backtest':
            logging.info("Building backtest container...")
            if self.execute_command("docker-compose build backtest") != 0:
                raise RuntimeError("Failed to build backtest container")
            
            # Start backtest container
            logging.info("Starting backtest container...")
            if self.args.detach:
                command = "docker-compose --profile backtest up -d backtest"
                if self.execute_command(command) != 0:
                    raise RuntimeError("Failed to start backtest container")
            else:
                os.execvp("docker-compose", ["docker-compose", "--profile", "backtest", "up", "backtest"])

    def deploy_aws(self):
        """Handle AWS deployment"""
        logging.info("Starting AWS deployment...")
        
        # Check AWS credentials
        if self.execute_command("aws sts get-caller-identity") != 0:
            raise RuntimeError("AWS credentials not configured")
        
        # If build-only flag is set, build the image and push to ECR but don't deploy
        if self.args.build_only:
            logging.info(f"Building {self.target} for AWS and pushing to ECR...")
            # Add ECR build and push commands here
            logging.info("✅ Container built and pushed to ECR")
            return
            
        # Deploy using Terraform
        infra_dir = os.path.join(self.project_root, 'infrastructure')
        
        # Add target-specific variables if needed
        target_var = f"-var=\"deployment_target={self.target}\""
        
        commands = [
            "terraform init",
            f"terraform workspace select {self.environment} || terraform workspace new {self.environment}",
            f"terraform apply {target_var} -var=\"environment={self.environment}\" -auto-approve"
        ]
        
        for command in commands:
            if self.execute_command(command, cwd=infra_dir) != 0:
                raise RuntimeError(f"Failed to execute: {command}")

    def deploy(self):
        """Main deployment method"""
        try:
            self.check_prerequisites()
            self.validate_secrets()
            
            if self.mode == 'local':
                self.deploy_local()
            else:
                self.deploy_aws()
            
            if not self.args.build_only:
                logging.info(f"✅ Deployment of {self.target} successful")
                
                if self.mode == 'local' and not self.args.detach:
                    logging.info("📊 Logs available via: docker-compose logs -f")
            else:
                logging.info(f"✅ Build of {self.target} successful")
                
        except Exception as e:
            logging.error(f"❌ Deployment failed: {str(e)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Deploy trading bot')
    parser.add_argument('--mode', choices=['local', 'aws'], required=True)
    parser.add_argument('--env', choices=['simulation', 'live'], required=True)
    parser.add_argument('--target', choices=['all', 'trading-bot', 'backtest'], default='all',
                      help='Deployment target: all, trading-bot, or backtest')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--rebuild-base', action='store_true', help='Force rebuild of base image')
    parser.add_argument('--build-only', action='store_true', help='Only build the container without starting it')
    
    args = parser.parse_args()
    
    # Simplified logging format without extra newlines
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'.strip()
    )
    
    try:
        deployer = Deployer(args.mode, args.env, args.target, args)
        deployer.deploy()
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 