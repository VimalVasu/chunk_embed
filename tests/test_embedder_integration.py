"""Integration tests for embedder with config system."""

import os
import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import tempfile

from src.config import get_config, OpenAIConfig
from src.embedder import EmbeddingClient, create_embedding_client


class TestEmbedderConfigIntegration:
    """Test embedder integration with config system."""
    
    def test_embedder_with_config_from_env(self):
        """Test embedder initialization with config loaded from environment."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key_from_env',
            'OPENAI_MODEL': 'text-embedding-ada-002',
            'DEFAULT_BATCH_SIZE': '15',
            'MAX_RETRIES': '2'
        }):
            with patch('src.embedder.OpenAI') as mock_openai:
                mock_openai.return_value = Mock()
                
                # Load config from environment - need to remove default config file
                with patch('pathlib.Path.exists', return_value=False):
                    config = get_config()
                
                # Create embedder with config
                client = EmbeddingClient(config.openai)
                
                # Verify OpenAI client was initialized correctly
                mock_openai.assert_called_once_with(
                    api_key='test_key_from_env',
                    timeout=30
                )
                
                # Verify configuration was loaded correctly
                assert client.config.api_key == 'test_key_from_env'
                assert client.config.model == 'text-embedding-ada-002'
                assert client.config.batch_size == 15
                assert client.config.max_retries == 2
    
    def test_embedder_with_config_from_file(self):
        """Test embedder initialization with config from JSON file."""
        # Create temporary config file
        config_data = {
            "openai": {
                "api_key": "test_key_from_file",
                "model": "text-embedding-ada-002",
                "batch_size": 25,
                "max_retries": 4,
                "timeout": 60
            },
            "chromadb": {
                "db_path": "./data/chroma_db",
                "collection_name": "test_collection",
                "distance_metric": "cosine"
            },
            "chunking": {
                "strategy": "fixed_window",
                "window_size": 60,
                "overlap_seconds": 5,
                "min_chunk_length": 10,
                "max_chunk_duration": 300,
                "speaker_change_threshold": 0.0,
                "merge_consecutive_same_speaker": True
            },
            "logging": {
                "level": "INFO",
                "file_path": "./logs/chunking_service.log",
                "max_file_size": 10000000,
                "backup_count": 5
            },
            "development": {
                "environment": "development",
                "debug": False,
                "fast_mode": False,
                "cache_embeddings": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            with patch('src.embedder.OpenAI') as mock_openai:
                mock_openai.return_value = Mock()
                
                # Load config from file
                config = get_config(config_file)
                
                # Create embedder with config
                client = EmbeddingClient(config.openai)
                
                # Verify OpenAI client was initialized correctly
                mock_openai.assert_called_once_with(
                    api_key='test_key_from_file',
                    timeout=60
                )
                
                # Verify configuration was loaded correctly
                assert client.config.api_key == 'test_key_from_file'
                assert client.config.model == 'text-embedding-ada-002'
                assert client.config.batch_size == 25
                assert client.config.max_retries == 4
                assert client.config.timeout == 60
                
        finally:
            os.unlink(config_file)
    
    def test_embedder_factory_with_config(self):
        """Test embedder factory function with config."""
        config = OpenAIConfig(
            api_key="test_factory_key",
            model="text-embedding-ada-002",
            batch_size=10,
            max_retries=3,
            timeout=30
        )
        
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            # Create real client
            client = create_embedding_client(config, use_mock=False)
            assert isinstance(client, EmbeddingClient)
            
            # Verify initialization
            mock_openai.assert_called_once_with(
                api_key='test_factory_key',
                timeout=30
            )
        
        # Create mock client
        mock_client = create_embedding_client(config, use_mock=True)
        assert mock_client.__class__.__name__ == 'MockEmbeddingClient'
    
    def test_embedder_config_integration(self):
        """Test embedder integration with config system."""
        # Test that embedder can be created with valid config
        config = OpenAIConfig(
            api_key="test_integration_key",
            model="text-embedding-ada-002",
            batch_size=20,
            max_retries=3,
            timeout=30
        )
        
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            # Create embedder with config
            client = EmbeddingClient(config)
            
            # Verify client was initialized properly
            mock_openai.assert_called_once_with(
                api_key='test_integration_key',
                timeout=30
            )
            
            # Verify configuration was properly set
            assert client.config.api_key == 'test_integration_key'
            assert client.config.model == 'text-embedding-ada-002'
            assert client.config.batch_size == 20
            assert client.config.max_retries == 3
            assert client.config.timeout == 30
    
    def test_embedder_missing_api_key_error(self):
        """Test error when API key is missing from config."""
        config = OpenAIConfig(
            api_key="",  # Empty API key
            model="text-embedding-ada-002"
        )
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingClient(config)
    
    def test_embedder_invalid_config_validation(self):
        """Test embedder with invalid configuration values."""
        # Test invalid batch size
        with pytest.raises(ValueError):
            OpenAIConfig(
                api_key="valid_key",
                model="text-embedding-ada-002",
                batch_size=0  # Invalid batch size
            )
        
        # Test invalid max_retries
        with pytest.raises(ValueError):
            OpenAIConfig(
                api_key="valid_key",
                model="text-embedding-ada-002",
                max_retries=-1  # Invalid max_retries
            )
        
        # Test invalid timeout
        with pytest.raises(ValueError):
            OpenAIConfig(
                api_key="valid_key",
                model="text-embedding-ada-002",
                timeout=1  # Too low timeout
            )