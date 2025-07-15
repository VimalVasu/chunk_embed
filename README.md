# Chunking & Embedding Service

A localhost development tool for transcript chunking and embedding generation using OpenAI and ChromaDB.

## Quick Setup

1. **Create virtual environment and install dependencies:**
   ```bash
   ./setup_env.sh
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env with your OpenAI API key
   ```

3. **Validate setup:**
   ```bash
   python setup_dev.py
   ```

## Project Structure

```
chunk_embed/
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   └── utils.py           # Utility functions
├── tests/                 # Test files
├── test_data/            # Sample transcript data
├── data/                 # ChromaDB storage (created automatically)
├── logs/                 # Log files (created automatically)
├── requirements.txt      # Python dependencies
├── .env.template        # Environment variables template
├── config.default.json  # Default configuration
└── setup_env.sh         # Environment setup script
```

## Configuration

The service supports multiple configuration methods with the following precedence:

1. **config.local.json** (highest priority, not tracked in git)
2. **Environment variables** from .env file
3. **config.default.json** (fallback defaults)

### Creating Local Configuration

```bash
# Copy template and customize
cp config.local.json.template config.local.json
# Edit config.local.json with your specific settings
```

### Environment Variables

Key environment variables (see `.env.template`):

- `OPENAI_API_KEY`: Your OpenAI API key
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENVIRONMENT`: development or production
- `DEBUG`: Enable debug mode (true/false)

## Development Workflow

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Validate environment:**
   ```bash
   python setup_dev.py
   ```

3. **Run tests:**
   ```bash
   pytest tests/
   ```

## Configuration Examples

### Development Mode (config.local.json)
```json
{
  "openai": {
    "api_key": "sk-your-actual-key",
    "batch_size": 10
  },
  "development": {
    "debug": true,
    "fast_mode": true
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

### Production Mode
```json
{
  "development": {
    "environment": "production",
    "debug": false,
    "fast_mode": false
  },
  "logging": {
    "level": "INFO"
  }
}
```

## Next Steps

After completing Task 1 setup, proceed with:
- Task 2: Test Data Creation
- Task 3: Transcript Loader Module
- Task 4: Basic Chunking Engine

## Troubleshooting

### Common Issues

1. **"No module named 'pydantic'"**
   - Run: `pip install -r requirements.txt`

2. **"OpenAI API key is required"**
   - Update your `.env` file or `config.local.json` with a valid API key

3. **"Missing required directory"**
   - Run: `mkdir -p src tests test_data data logs`

### Getting Help

Run the setup validation script for detailed diagnostics:
```bash
python setup_dev.py
```