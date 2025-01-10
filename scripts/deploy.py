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
    def __init__(self, mode: str, environment: str, args):
        self.mode = mode  # 'local' or 'aws'
        self.environment = environment  # 'simulation' or 'production'
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
                raise RuntimeError("secrets.json is required for production deployment")
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

    def deploy_local(self):
        """Handle local deployment"""
        logging.info("Starting local deployment...")
        
        # Build Docker containers
        logging.info("Building Docker containers...")
        if self.execute_command("docker-compose build --no-cache") != 0:
            raise RuntimeError("Failed to build Docker containers")
            
        # Start containers
        logging.info("Starting Docker containers...")
        command = "docker-compose up"
        if self.args.detach:
            command += " -d"
            
        if self.execute_command(command) != 0:
            raise RuntimeError("Failed to start Docker containers")
            
        # Verify containers are running
        if self.execute_command("docker-compose ps -q") != 0:
            raise RuntimeError("Container failed to start")

    def deploy_aws(self):
        """Handle AWS deployment"""
        logging.info("Starting AWS deployment...")
        
        # Check AWS credentials
        if self.execute_command("aws sts get-caller-identity") != 0:
            raise RuntimeError("AWS credentials not configured")
            
        # Deploy using Terraform
        infra_dir = os.path.join(self.project_root, 'infrastructure')
        
        commands = [
            "terraform init",
            f"terraform workspace select {self.environment} || terraform workspace new {self.environment}",
            f"terraform apply -var=\"environment={self.environment}\" -auto-approve"
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
                
            logging.info("✅ Deployment successful")
            
            if self.mode == 'local' and not self.args.detach:
                logging.info("📊 Logs available via: docker-compose logs -f")
                
        except Exception as e:
            logging.error(f"❌ Deployment failed: {str(e)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Deploy trading bot')
    parser.add_argument('--mode', choices=['local', 'aws'], required=True)
    parser.add_argument('--env', choices=['simulation', 'production'], required=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--detach', action='store_true')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        deployer = Deployer(args.mode, args.env, args)
        deployer.deploy()
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 