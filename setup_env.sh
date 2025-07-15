#!/bin/bash
# Setup script for localhost development

echo "Setting up Chunking & Embedding Service development environment..."

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"