#!/usr/bin/env python3
"""Development environment setup and validation script."""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils import setup_development_environment
    
    if __name__ == "__main__":
        success = setup_development_environment()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install dependencies first:")
    print("  python -m pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Setup failed: {e}")
    sys.exit(1)