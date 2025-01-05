#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import json
import logging
from typing import Optional
import shutil
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class ProgressIndicator:
    def __init__(self, message=""):
        self.message = message
        self.running = False
        self.thread = None

    def show_progress(self):
        while self.running:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(3)

    def start(self, message=None):
        if message:
            self.message = message
            sys.stdout.write(f"\n{message}")
            sys.stdout.flush()
        self.running = True
        self.thread = threading.Thread(target=self.show_progress)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\n")
        sys.stdout.flush()

class DeploymentError(Exception):
    """Custom exception for deployment errors"""
    pass

class Deployer:
    def __init__(self, mode: str, environment: str):
        self.mode = mode  # 'local' or 'aws'
        self.environment = environment  # 'simulation' or 'production'
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.progress = ProgressIndicator()

    def check_prerequisites(self) -> None:
        """Verify all required tools and configurations are present"""
        logger.info("Checking prerequisites...")
        
        try:
            # Check Docker for local deployment
            if self.mode == 'local':
                self._run_command(['docker', '--version'])
                self._run_command(['docker-compose', '--version'])
                
            # Check AWS CLI and Terraform for AWS deployment
            elif self.mode == 'aws':
                self._run_command(['aws', '--version'])
                self._run_command(['terraform', '--version'])
                
                # Verify AWS credentials
                self._run_command(['aws', 'sts', 'get-caller-identity'])
                
            logger.info("✅ All prerequisites checked successfully")
            
        except subprocess.CalledProcessError as e:
            raise DeploymentError(f"Prerequisites check failed: {str(e)}")

    def validate_secrets(self) -> None:
        """Validate presence and format of secrets"""
        logger.info("Validating secrets...")
        
        if self.mode == 'local':
            secrets_path = os.path.join(self.project_root, 'secrets.json')
            if not os.path.exists(secrets_path):
                raise DeploymentError("secrets.json not found")
            
            try:
                with open(secrets_path) as f:
                    secrets = json.load(f)
                    required_keys = ['api_key', 'api_secret', 'passphrase']
                    if not all(key in secrets for key in required_keys):
                        raise DeploymentError("Missing required keys in secrets.json")
            except json.JSONDecodeError:
                raise DeploymentError("Invalid JSON in secrets.json")
                
        elif self.mode == 'aws':
            # Check if AWS Secrets Manager contains required secrets
            try:
                self._run_command([
                    'aws', 'secretsmanager', 'describe-secret',
                    f'--secret-id', f'trading-bot/{self.environment}/coinbase-credentials'
                ])
            except subprocess.CalledProcessError:
                raise DeploymentError("AWS Secrets Manager credentials not configured")
        
        logger.info("✅ Secrets validation completed")

    def deploy_local(self) -> None:
        """Handle local deployment using Docker"""
        logger.info("Starting local deployment...")
        
        try:
            # Build and start containers
            logger.info("Building Docker containers...")
            os.chdir(self.project_root)
            
            self._run_command([
                'docker-compose',
                'build',
                '--no-cache'
            ])
            
            up_command = ['docker-compose', 'up']
            if self.args.detach:
                up_command.append('-d')
            
            logger.info("Starting Docker containers...")
            self._run_command(up_command)
            
            # Verify container is running
            result = self._run_command([
                'docker-compose',
                'ps',
                '-q'
            ])
            
            if not result.strip():
                raise DeploymentError("Container failed to start")
            
            logger.info("✅ Local deployment successful")
            logger.info("📊 Logs available via: docker-compose logs -f")
            
        except subprocess.CalledProcessError as e:
            raise DeploymentError(f"Local deployment failed: {str(e)}")
        finally:
            # Change back to original directory
            os.chdir(self.project_root)

    def deploy_aws(self) -> None:
        """Handle AWS deployment using Terraform"""
        logger.info("Starting AWS deployment...")
        
        try:
            # Create deployment package
            self._create_deployment_package()
            
            # Initialize and apply Terraform
            os.chdir(os.path.join(self.project_root, 'infrastructure'))
            
            self._run_command(['terraform', 'init'])
            
            # Plan with variables
            self._run_command([
                'terraform', 'plan',
                '-var', f'trading_mode={self.environment}',
                '-out=tfplan'
            ])
            
            # Apply the plan
            self._run_command(['terraform', 'apply', 'tfplan'])
            
            logger.info("✅ AWS deployment successful")
            logger.info("📊 CloudWatch Logs available in AWS Console")
            
        except subprocess.CalledProcessError as e:
            raise DeploymentError(f"AWS deployment failed: {str(e)}")

    def _create_deployment_package(self) -> None:
        """Create ZIP package for Lambda deployment"""
        logger.info("Creating deployment package...")
        
        deployment_dir = os.path.join(self.project_root, 'deployment')
        os.makedirs(deployment_dir, exist_ok=True)
        
        zip_path = os.path.join(deployment_dir, 'lambda.zip')
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        # Create virtual environment and install dependencies
        venv_path = os.path.join(deployment_dir, 'venv')
        self._run_command(['python', '-m', 'venv', venv_path])
        
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        self._run_command([pip_path, 'install', '-r', 
                          os.path.join(self.project_root, 'requirements.txt')])
        
        # Create ZIP file
        shutil.make_archive(
            os.path.join(deployment_dir, 'lambda'),
            'zip',
            os.path.join(self.project_root, 'src')
        )
        
        logger.info("✅ Deployment package created successfully")

    def _run_command(self, command: list) -> str:
        """Run a shell command and return its output"""
        logger.info(f"\n🔧 Executing: {' '.join(command)}")
        logger.info(f"📍 Working directory: {os.getcwd()}")
        
        # For Docker Compose commands, stream output in real-time
        if 'docker-compose' in command:
            self.progress.start("Waiting for Docker Compose")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            output = []
            for line in process.stdout:
                line = line.rstrip()
                self.progress.stop()
                logger.info(line)
                self.progress.start("")
                output.append(line)
            
            process.wait()
            self.progress.stop()
            logger.info(f"📋 Exit code: {process.returncode}")
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            return '\n'.join(output)
        else:
            self.progress.start(f"Running {command[0]}")
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            self.progress.stop()
            logger.info(f"📋 Exit code: {result.returncode}")
            if result.stdout:
                logger.info(f"📤 Output:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"⚠️ Stderr:\n{result.stderr}")
            return result.stdout

def main():
    parser = argparse.ArgumentParser(description='Deploy trading bot locally or to AWS')
    parser.add_argument('--mode', choices=['local', 'aws'], required=True,
                      help='Deployment mode')
    parser.add_argument('--env', choices=['simulation', 'production'], required=True,
                      help='Environment to deploy to')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--detach', '-d', action='store_true',
                      help='Run containers in detached mode')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        deployer = Deployer(args.mode, args.env)
        deployer.check_prerequisites()
        deployer.validate_secrets()
        
        if args.mode == 'local':
            deployer.deploy_local()
        else:
            deployer.deploy_aws()
            
    except DeploymentError as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 