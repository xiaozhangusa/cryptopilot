from setuptools import setup, find_packages

setup(
    name="cryptopilot",
    version="0.1",
    packages=find_packages(include=['src', 'src.*', 'bot_strategy', 'bot_strategy.*']),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "requests>=2.26.0",
        "boto3>=1.24.0",
        "python-dotenv>=0.19.0"
    ],
) 