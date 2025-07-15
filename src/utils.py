"""Utility functions for configuration and environment management."""

import os
import json
from pathlib import Path
from typing import Dict, Any
from .config import AppConfig, get_config


def validate_environment_setup() -> Dict[str, Any]:
    """Validate that the environment is properly set up for development."""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    # Check for required directories
    required_dirs = ["src", "tests", "test_data", "data", "logs"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            validation_results["errors"].append(f"Missing required directory: {dir_name}")
            validation_results["valid"] = False
    
    # Check for .env file
    if not Path(".env").exists():
        if Path(".env.template").exists():
            validation_results["warnings"].append(
                "No .env file found. Copy .env.template to .env and configure your API keys."
            )
        else:
            validation_results["errors"].append("No .env file or .env.template found")
            validation_results["valid"] = False
    
    # Check for virtual environment
    if not Path("venv").exists():
        validation_results["warnings"].append(
            "No virtual environment found. Run: python -m venv venv"
        )
    
    # Check configuration loading
    try:
        config = get_config()
        validation_results["info"].append(f"Configuration loaded successfully: {config.development.environment} mode")
        
        # Check API key
        if config.openai.api_key.startswith("your_"):
            validation_results["warnings"].append(
                "OpenAI API key appears to be a placeholder. Update your .env or config.local.json file."
            )
    except Exception as e:
        validation_results["errors"].append(f"Configuration loading failed: {str(e)}")
        validation_results["valid"] = False
    
    return validation_results


def create_local_config_from_template():
    """Create a local config file from template if it doesn't exist."""
    local_config_path = Path("config.local.json")
    template_path = Path("config.local.json.template")
    
    if local_config_path.exists():
        print("config.local.json already exists")
        return
    
    if not template_path.exists():
        print("config.local.json.template not found")
        return
    
    # Copy template to local config
    with open(template_path, "r") as f:
        template_content = json.load(f)
    
    # Remove comment field
    if "_comment" in template_content:
        del template_content["_comment"]
    
    with open(local_config_path, "w") as f:
        json.dump(template_content, f, indent=2)
    
    print(f"Created {local_config_path} from template")
    print("Please update the API key and other settings as needed")


def get_environment_info() -> Dict[str, str]:
    """Get information about the current environment."""
    config = get_config()
    
    return {
        "environment": config.development.environment,
        "debug": str(config.development.debug),
        "fast_mode": str(config.development.fast_mode),
        "log_level": config.logging.level,
        "chunk_strategy": config.chunking.strategy,
        "chunk_window": str(config.chunking.window_size),
        "batch_size": str(config.openai.batch_size),
        "db_path": config.chromadb.db_path,
    }


def setup_development_environment():
    """Interactive setup for development environment."""
    print("=== Chunking & Embedding Service - Development Setup ===\n")
    
    # Validate current setup
    validation = validate_environment_setup()
    
    if validation["errors"]:
        print("❌ Setup Issues Found:")
        for error in validation["errors"]:
            print(f"  - {error}")
        print()
    
    if validation["warnings"]:
        print("⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
        print()
    
    if validation["info"]:
        print("ℹ️  Information:")
        for info in validation["info"]:
            print(f"  - {info}")
        print()
    
    # Offer to create local config
    if not Path("config.local.json").exists():
        response = input("Create config.local.json from template? (y/n): ")
        if response.lower() == 'y':
            create_local_config_from_template()
            print()
    
    # Show environment info
    print("📋 Current Environment Settings:")
    env_info = get_environment_info()
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    print()
    
    if validation["valid"]:
        print("✅ Environment setup is valid!")
    else:
        print("❌ Please fix the issues above before proceeding.")
    
    return validation["valid"]


if __name__ == "__main__":
    setup_development_environment()