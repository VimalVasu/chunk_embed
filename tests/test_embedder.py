"""Comprehensive tests for the embedder module."""

import pytest
import time
import threading
import unittest.mock
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from src.embedder import (
    EmbeddingClient,
    MockEmbeddingClient,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingMetadata,
    RateLimiter,
    create_embedding_client
)
from src.config import OpenAIConfig


class TestEmbeddingRequest:
    """Test EmbeddingRequest dataclass."""
    
    def test_embedding_request_creation(self):
        """Test EmbeddingRequest creation with required fields."""
        request = EmbeddingRequest(
            text="Hello world",
            chunk_id="chunk_1"
        )
        
        assert request.text == "Hello world"
        assert request.chunk_id == "chunk_1"
        assert request.metadata == {}
    
    def test_embedding_request_with_metadata(self):
        """Test EmbeddingRequest creation with metadata."""
        metadata = {"speaker": "john", "timestamp": 123.45}
        request = EmbeddingRequest(
            text="Hello world",
            chunk_id="chunk_1",
            metadata=metadata
        )
        
        assert request.metadata == metadata


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""
    
    def test_embedding_result_creation(self):
        """Test EmbeddingResult creation."""
        embedding = [0.1, 0.2, 0.3]
        created_at = datetime.now()
        
        result = EmbeddingResult(
            chunk_id="chunk_1",
            embedding=embedding,
            model="text-embedding-ada-002",
            created_at=created_at
        )
        
        assert result.chunk_id == "chunk_1"
        assert result.embedding == embedding
        assert result.model == "text-embedding-ada-002"
        assert result.created_at == created_at
        assert result.metadata == {}


class TestEmbeddingMetadata:
    """Test EmbeddingMetadata Pydantic model."""
    
    def test_embedding_metadata_validation(self):
        """Test EmbeddingMetadata validation."""
        metadata = EmbeddingMetadata(
            model="text-embedding-ada-002",
            model_version="002",
            created_at=datetime.now(),
            processing_time=1.23,
            embedding_dimension=1536
        )
        
        assert metadata.model == "text-embedding-ada-002"
        assert metadata.model_version == "002"
        assert metadata.embedding_version == "1.0"  # Default value
        assert metadata.processing_time == 1.23
        assert metadata.embedding_dimension == 1536
        assert metadata.request_id is None
    
    def test_embedding_metadata_with_request_id(self):
        """Test EmbeddingMetadata with request ID."""
        metadata = EmbeddingMetadata(
            model="text-embedding-ada-002",
            model_version="002",
            created_at=datetime.now(),
            processing_time=1.23,
            embedding_dimension=1536,
            request_id="req_123"
        )
        
        assert metadata.request_id == "req_123"
    
    def test_embedding_metadata_comprehensive(self):
        """Test comprehensive EmbeddingMetadata with all fields."""
        metadata = EmbeddingMetadata(
            model="text-embedding-ada-002",
            model_version="002",
            embedding_version="1.0",
            created_at=datetime.now(),
            request_id="req_123",
            processing_time=1.23,
            embedding_dimension=1536,
            batch_size=10,
            retry_count=2,
            openai_api_version="v1",
            client_version="1.0",
            input_tokens=100,
            fingerprint="text-embedding-ada-002-20240717"
        )
        
        assert metadata.batch_size == 10
        assert metadata.retry_count == 2
        assert metadata.openai_api_version == "v1"
        assert metadata.client_version == "1.0"
        assert metadata.input_tokens == 100
        assert metadata.fingerprint == "text-embedding-ada-002-20240717"


class TestEmbeddingClient:
    """Test EmbeddingClient class."""
    
    @pytest.fixture
    def valid_config(self):
        """Create valid OpenAI config for testing."""
        return OpenAIConfig(
            api_key="test_api_key",
            model="text-embedding-ada-002",
            batch_size=10,
            max_retries=2,
            timeout=30
        )
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        return mock_client
    
    def test_client_initialization_success(self, valid_config):
        """Test successful client initialization."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            client = EmbeddingClient(valid_config)
            
            assert client.config == valid_config
            assert client.client is not None
            mock_openai.assert_called_once_with(
                api_key="test_api_key",
                timeout=30
            )
    
    def test_client_initialization_empty_api_key(self):
        """Test client initialization with empty API key."""
        config = OpenAIConfig(
            api_key="",
            model="text-embedding-ada-002"
        )
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingClient(config)
    
    def test_client_initialization_invalid_api_key(self):
        """Test client initialization with invalid API key."""
        config = OpenAIConfig(
            api_key="your_openai_api_key_here",
            model="text-embedding-ada-002"
        )
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingClient(config)
    
    def test_client_setup_with_env_api_key(self, valid_config):
        """Test client setup with API key from environment."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            client = EmbeddingClient(valid_config)
            
            # Verify OpenAI client was created with correct parameters
            mock_openai.assert_called_once_with(
                api_key="test_api_key",
                timeout=30
            )
            assert client.client is not None
    
    def test_client_setup_timeout_configuration(self, valid_config):
        """Test client setup with timeout configuration."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            client = EmbeddingClient(valid_config)
            
            # Check that timeout is properly passed to OpenAI client
            args, kwargs = mock_openai.call_args
            assert kwargs['timeout'] == 30
    
    def test_client_setup_api_key_validation(self):
        """Test that API key validation works correctly."""
        # Test with empty string
        config = OpenAIConfig(api_key="", model="text-embedding-ada-002")
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingClient(config)
        
        # Test with placeholder key
        config = OpenAIConfig(api_key="your_api_key", model="text-embedding-ada-002")
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingClient(config)
    
    def test_generate_embedding_success(self, valid_config, mock_openai_client):
        """Test successful embedding generation."""
        with patch('src.embedder.OpenAI', return_value=mock_openai_client):
            client = EmbeddingClient(valid_config)
            
            result = client.generate_embedding("Hello world", "chunk_1")
            
            assert result.chunk_id == "chunk_1"
            assert result.embedding == [0.1, 0.2, 0.3]
            assert result.model == "text-embedding-ada-002"
            assert isinstance(result.created_at, datetime)
            assert result.metadata == {}
            assert result.version == "1.0"
            assert result.embedding_metadata is not None
            assert result.embedding_metadata.model == "text-embedding-ada-002"
            assert result.embedding_metadata.model_version == "002"
            assert result.embedding_metadata.embedding_dimension == 1536
    
    def test_generate_embedding_with_metadata(self, valid_config, mock_openai_client):
        """Test embedding generation with metadata."""
        with patch('src.embedder.OpenAI', return_value=mock_openai_client):
            client = EmbeddingClient(valid_config)
            metadata = {"speaker": "john"}
            
            result = client.generate_embedding("Hello world", "chunk_1", metadata)
            
            assert result.metadata == metadata
    
    def test_generate_embedding_empty_text(self, valid_config):
        """Test embedding generation with empty text."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            with pytest.raises(ValueError, match="Text cannot be empty"):
                client.generate_embedding("", "chunk_1")
    
    def test_generate_embedding_whitespace_text(self, valid_config):
        """Test embedding generation with whitespace-only text."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            with pytest.raises(ValueError, match="Text cannot be empty"):
                client.generate_embedding("   ", "chunk_1")
    
    def test_generate_embeddings_batch_success(self, valid_config, mock_openai_client):
        """Test successful batch embedding generation."""
        mock_openai_client.embeddings.create.return_value.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        with patch('src.embedder.OpenAI', return_value=mock_openai_client):
            client = EmbeddingClient(valid_config)
            
            requests = [
                EmbeddingRequest("Hello", "chunk_1"),
                EmbeddingRequest("World", "chunk_2")
            ]
            
            results = client.generate_embeddings_batch(requests)
            
            assert len(results) == 2
            assert results[0].chunk_id == "chunk_1"
            assert results[0].embedding == [0.1, 0.2, 0.3]
            assert results[1].chunk_id == "chunk_2"
            assert results[1].embedding == [0.4, 0.5, 0.6]
            
            # Check batch metadata
            assert results[0].embedding_metadata.batch_size == 2
            assert results[1].embedding_metadata.batch_size == 2
            assert results[0].version == "1.0"
            assert results[1].version == "1.0"
    
    def test_generate_embeddings_batch_empty(self, valid_config):
        """Test batch embedding generation with empty list."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            results = client.generate_embeddings_batch([])
            
            assert results == []
    
    def test_split_into_batches(self, valid_config):
        """Test batch splitting functionality."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            requests = [
                EmbeddingRequest(f"Text {i}", f"chunk_{i}")
                for i in range(25)
            ]
            
            batches = client._split_into_batches(requests)
            
            assert len(batches) == 3  # 25 items with batch_size=10
            assert len(batches[0]) == 10
            assert len(batches[1]) == 10
            assert len(batches[2]) == 5
    
    def test_validate_embedding_valid(self, valid_config):
        """Test embedding validation with valid embedding."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            embedding = [0.1, 0.2, 0.3, 0.4]
            assert client.validate_embedding(embedding) is True
    
    def test_validate_embedding_invalid_type(self, valid_config):
        """Test embedding validation with invalid type."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            assert client.validate_embedding("not_a_list") is False
    
    def test_validate_embedding_empty(self, valid_config):
        """Test embedding validation with empty list."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            assert client.validate_embedding([]) is False
    
    def test_validate_embedding_invalid_values(self, valid_config):
        """Test embedding validation with invalid values."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            assert client.validate_embedding([0.1, "invalid", 0.3]) is False
    
    def test_get_model_info(self, valid_config):
        """Test model information retrieval."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            info = client.get_model_info()
            
            assert info["model"] == "text-embedding-ada-002"
            assert info["batch_size"] == 10
            assert info["max_retries"] == 2
            assert info["timeout"] == 30
    
    def test_retry_logic_success_after_failure(self, valid_config):
        """Test retry logic with success after failure."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # First call fails, second succeeds
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.side_effect = [
                Exception("API Error"),
                mock_response
            ]
            
            client = EmbeddingClient(valid_config)
            
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = client.generate_embedding("Hello", "chunk_1")
                
                assert result.embedding == [0.1, 0.2, 0.3]
                assert mock_client.embeddings.create.call_count == 2
    
    def test_retry_logic_max_retries_exceeded(self, valid_config):
        """Test retry logic with max retries exceeded."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # All calls fail
            mock_client.embeddings.create.side_effect = Exception("API Error")
            
            client = EmbeddingClient(valid_config)
            
            with patch('time.sleep'):  # Mock sleep to speed up test
                with pytest.raises(Exception, match="API Error"):
                    client.generate_embedding("Hello", "chunk_1")
                
                # Should try max_retries + 1 times
                assert mock_client.embeddings.create.call_count == 3
    
    def test_embedding_metadata_tracking(self, valid_config, mock_openai_client):
        """Test embedding metadata tracking functionality."""
        with patch('src.embedder.OpenAI', return_value=mock_openai_client):
            client = EmbeddingClient(valid_config)
            
            result = client.generate_embedding("Hello world", "chunk_1")
            
            # Check that metadata is properly tracked
            assert result.embedding_metadata is not None
            metadata = result.embedding_metadata
            
            assert metadata.model == "text-embedding-ada-002"
            assert metadata.model_version == "002"
            assert metadata.embedding_version == "1.0"
            assert metadata.embedding_dimension == 1536
            assert metadata.batch_size == 1
            assert metadata.retry_count == 0
            assert metadata.openai_api_version == "v1"
            assert metadata.client_version == "1.0"
            assert metadata.input_tokens > 0  # Should estimate tokens
            assert metadata.fingerprint is not None
            assert metadata.processing_time > 0
    
    def test_embedding_metadata_with_retries(self, valid_config):
        """Test embedding metadata tracks retry count."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # First call fails, second succeeds
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.side_effect = [
                Exception("API Error"),
                mock_response
            ]
            
            client = EmbeddingClient(valid_config)
            
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = client.generate_embedding("Hello", "chunk_1")
                
                # Check that retry count is tracked
                assert result.embedding_metadata.retry_count == 1
    
    def test_embedding_versioning_consistency(self, valid_config, mock_openai_client):
        """Test that versioning is consistent across embeddings."""
        with patch('src.embedder.OpenAI', return_value=mock_openai_client):
            client = EmbeddingClient(valid_config)
            
            result1 = client.generate_embedding("Hello world", "chunk_1")
            result2 = client.generate_embedding("Different text", "chunk_2")
            
            # Both should have same version and model metadata
            assert result1.version == result2.version
            assert result1.embedding_metadata.model_version == result2.embedding_metadata.model_version
            assert result1.embedding_metadata.embedding_version == result2.embedding_metadata.embedding_version
            assert result1.embedding_metadata.client_version == result2.embedding_metadata.client_version
    
    def test_token_estimation(self, valid_config, mock_openai_client):
        """Test token estimation in metadata."""
        with patch('src.embedder.OpenAI', return_value=mock_openai_client):
            client = EmbeddingClient(valid_config)
            
            short_text = "Hi"
            long_text = "This is a much longer text that should have more tokens"
            
            result1 = client.generate_embedding(short_text, "chunk_1")
            result2 = client.generate_embedding(long_text, "chunk_2")
            
            # Longer text should have more estimated tokens
            assert result2.embedding_metadata.input_tokens > result1.embedding_metadata.input_tokens
    
    def test_retry_logic_exponential_backoff(self, valid_config):
        """Test exponential backoff timing in retry logic."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # All calls fail to test backoff
            mock_client.embeddings.create.side_effect = Exception("API Error")
            
            client = EmbeddingClient(valid_config)
            
            # Mock time.sleep to track backoff timing
            with patch('time.sleep') as mock_sleep:
                with pytest.raises(Exception, match="API Error"):
                    client.generate_embedding("Hello", "chunk_1")
                
                # Should have called sleep with exponential backoff: 1s, 2s
                expected_calls = [unittest.mock.call(1), unittest.mock.call(2)]
                mock_sleep.assert_has_calls(expected_calls)
    
    def test_retry_logic_different_exceptions(self, valid_config):
        """Test retry logic with different types of exceptions."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Test with different exception types
            exceptions = [
                Exception("Network error"),
                ConnectionError("Connection failed"),
                TimeoutError("Request timed out"),
                ValueError("Invalid request")
            ]
            
            for exception in exceptions:
                mock_client.embeddings.create.side_effect = exception
                client = EmbeddingClient(valid_config)
                
                with patch('time.sleep'):
                    with pytest.raises(type(exception)):
                        client.generate_embedding("Hello", "chunk_1")
                
                # Should try max_retries + 1 times
                assert mock_client.embeddings.create.call_count == 3
                mock_client.embeddings.create.reset_mock()
    
    def test_retry_logic_batch_processing(self, valid_config):
        """Test retry logic works correctly with batch processing."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # First call fails, second succeeds
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6])
            ]
            mock_client.embeddings.create.side_effect = [
                Exception("API Error"),
                mock_response
            ]
            
            client = EmbeddingClient(valid_config)
            
            requests = [
                EmbeddingRequest("Hello", "chunk_1"),
                EmbeddingRequest("World", "chunk_2")
            ]
            
            with patch('time.sleep'):
                results = client.generate_embeddings_batch(requests)
                
                assert len(results) == 2
                assert results[0].embedding_metadata.retry_count == 1
                assert results[1].embedding_metadata.retry_count == 1
    
    def test_retry_logic_immediate_success(self, valid_config, mock_openai_client):
        """Test that retry count is 0 when API succeeds immediately."""
        with patch('src.embedder.OpenAI', return_value=mock_openai_client):
            client = EmbeddingClient(valid_config)
            
            result = client.generate_embedding("Hello", "chunk_1")
            
            # Should have zero retries on immediate success
            assert result.embedding_metadata.retry_count == 0
    
    def test_retry_logic_max_retries_configuration(self):
        """Test retry logic respects max_retries configuration."""
        config = OpenAIConfig(
            api_key="test_api_key",
            model="text-embedding-ada-002",
            max_retries=1  # Only 1 retry
        )
        
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # All calls fail
            mock_client.embeddings.create.side_effect = Exception("API Error")
            
            client = EmbeddingClient(config)
            
            with patch('time.sleep'):
                with pytest.raises(Exception, match="API Error"):
                    client.generate_embedding("Hello", "chunk_1")
                
                # Should try max_retries + 1 times (1 + 1 = 2 total attempts)
                assert mock_client.embeddings.create.call_count == 2


class TestMockEmbeddingClient:
    """Test MockEmbeddingClient class."""
    
    @pytest.fixture
    def valid_config(self):
        """Create valid OpenAI config for testing."""
        return OpenAIConfig(
            api_key="test_api_key",
            model="text-embedding-ada-002",
            batch_size=10,
            max_retries=2,
            timeout=30
        )
    
    def test_mock_client_initialization(self, valid_config):
        """Test mock client initialization."""
        client = MockEmbeddingClient(valid_config)
        
        assert client.config == valid_config
        assert client.client is None
    
    def test_mock_generate_embedding(self, valid_config):
        """Test mock embedding generation."""
        client = MockEmbeddingClient(valid_config)
        
        result = client.generate_embedding("Hello world", "chunk_1")
        
        assert result.chunk_id == "chunk_1"
        assert len(result.embedding) == 1536  # Standard OpenAI embedding size
        assert all(isinstance(x, float) for x in result.embedding)
        assert result.model == "text-embedding-ada-002"
        assert result.version == "1.0"
        assert result.embedding_metadata is not None
        assert result.embedding_metadata.retry_count == 0  # Mock never retries
    
    def test_mock_generate_embedding_deterministic(self, valid_config):
        """Test mock embedding generation is deterministic."""
        client = MockEmbeddingClient(valid_config)
        
        result1 = client.generate_embedding("Hello world", "chunk_1")
        result2 = client.generate_embedding("Hello world", "chunk_1")
        
        assert result1.embedding == result2.embedding
    
    def test_mock_generate_embedding_different_texts(self, valid_config):
        """Test mock embedding generation with different texts."""
        client = MockEmbeddingClient(valid_config)
        
        result1 = client.generate_embedding("Hello world", "chunk_1")
        result2 = client.generate_embedding("Different text", "chunk_2")
        
        assert result1.embedding != result2.embedding
    
    def test_mock_generate_embeddings_batch(self, valid_config):
        """Test mock batch embedding generation."""
        client = MockEmbeddingClient(valid_config)
        
        requests = [
            EmbeddingRequest("Hello", "chunk_1"),
            EmbeddingRequest("World", "chunk_2")
        ]
        
        results = client.generate_embeddings_batch(requests)
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[1].chunk_id == "chunk_2"
        assert len(results[0].embedding) == 1536
        assert len(results[1].embedding) == 1536
        
        # Check batch metadata
        assert results[0].embedding_metadata.batch_size == 2
        assert results[1].embedding_metadata.batch_size == 2


class TestCreateEmbeddingClient:
    """Test create_embedding_client factory function."""
    
    @pytest.fixture
    def valid_config(self):
        """Create valid OpenAI config for testing."""
        return OpenAIConfig(
            api_key="test_api_key",
            model="text-embedding-ada-002"
        )
    
    def test_create_real_client(self, valid_config):
        """Test creating real embedding client."""
        with patch('src.embedder.OpenAI'):
            client = create_embedding_client(valid_config, use_mock=False)
            
            assert isinstance(client, EmbeddingClient)
            assert not isinstance(client, MockEmbeddingClient)
    
    def test_create_mock_client(self, valid_config):
        """Test creating mock embedding client."""
        client = create_embedding_client(valid_config, use_mock=True)
        
        assert isinstance(client, MockEmbeddingClient)


class TestEmbeddingClientIntegration:
    """Integration tests for EmbeddingClient."""
    
    @pytest.fixture
    def valid_config(self):
        """Create valid OpenAI config for testing."""
        return OpenAIConfig(
            api_key="test_api_key",
            model="text-embedding-ada-002",
            batch_size=2,
            max_retries=1,
            timeout=30
        )
    
    def test_full_workflow_with_mock(self, valid_config):
        """Test full workflow with mock client."""
        client = MockEmbeddingClient(valid_config)
        
        # Test single embedding
        result = client.generate_embedding("Hello world", "chunk_1")
        assert client.validate_embedding(result.embedding)
        
        # Test batch embeddings
        requests = [
            EmbeddingRequest("First text", "chunk_1"),
            EmbeddingRequest("Second text", "chunk_2"),
            EmbeddingRequest("Third text", "chunk_3")
        ]
        
        batch_results = client.generate_embeddings_batch(requests)
        assert len(batch_results) == 3
        
        for result in batch_results:
            assert client.validate_embedding(result.embedding)
    
    def test_error_handling_integration(self, valid_config):
        """Test error handling across different methods."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            # Test empty text error
            with pytest.raises(ValueError):
                client.generate_embedding("", "chunk_1")
            
            # Test validation methods
            assert client.validate_embedding([0.1, 0.2, 0.3])
            assert not client.validate_embedding([])
            assert not client.validate_embedding("invalid")
            
            # Test model info
            info = client.get_model_info()
            assert "model" in info
            assert "batch_size" in info


class TestEmbeddingValidation:
    """Test embedding validation and error handling."""
    
    @pytest.fixture
    def valid_config(self):
        """Create valid OpenAI config for testing."""
        return OpenAIConfig(
            api_key="test_api_key",
            model="text-embedding-ada-002",
            batch_size=10,
            max_retries=2,
            timeout=30
        )
    
    @pytest.fixture
    def valid_embedding_result(self):
        """Create a valid embedding result for testing."""
        return EmbeddingResult(
            chunk_id="chunk_1",
            embedding=[0.1] * 1536,  # Valid 1536-dimensional embedding
            model="text-embedding-ada-002",
            created_at=datetime.now(),
            metadata={"test": "data"},
            version="1.0",
            embedding_metadata=EmbeddingMetadata(
                model="text-embedding-ada-002",
                model_version="002",
                created_at=datetime.now(),
                processing_time=0.5,
                embedding_dimension=1536
            )
        )
    
    def test_validate_embedding_dimensions_correct(self, valid_config):
        """Test embedding dimension validation with correct dimensions."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            embedding = [0.1] * 1536
            assert client.validate_embedding_dimensions(embedding) is True
    
    def test_validate_embedding_dimensions_incorrect(self, valid_config):
        """Test embedding dimension validation with incorrect dimensions."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            embedding = [0.1] * 512  # Wrong dimension
            assert client.validate_embedding_dimensions(embedding) is False
    
    def test_validate_embedding_dimensions_custom(self, valid_config):
        """Test embedding dimension validation with custom expected dimension."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            embedding = [0.1] * 512
            assert client.validate_embedding_dimensions(embedding, expected_dim=512) is True
    
    def test_validate_embedding_values_valid(self, valid_config):
        """Test embedding value validation with valid values."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            embedding = [0.1, -0.5, 0.0, 1.5, -2.0]
            assert client.validate_embedding_values(embedding) is True
    
    def test_validate_embedding_values_nan(self, valid_config):
        """Test embedding value validation with NaN values."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            embedding = [0.1, float('nan'), 0.3]
            assert client.validate_embedding_values(embedding) is False
    
    def test_validate_embedding_values_infinity(self, valid_config):
        """Test embedding value validation with infinity values."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            # Test positive infinity
            embedding = [0.1, float('inf'), 0.3]
            assert client.validate_embedding_values(embedding) is False
            
            # Test negative infinity
            embedding = [0.1, float('-inf'), 0.3]
            assert client.validate_embedding_values(embedding) is False
    
    def test_validate_embedding_values_extreme(self, valid_config):
        """Test embedding value validation with extreme values."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            # Test extreme positive value
            embedding = [0.1, 101.0, 0.3]
            assert client.validate_embedding_values(embedding) is False
            
            # Test extreme negative value
            embedding = [0.1, -101.0, 0.3]
            assert client.validate_embedding_values(embedding) is False
    
    def test_validate_embedding_result_valid(self, valid_config, valid_embedding_result):
        """Test comprehensive embedding result validation with valid result."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is True
            assert len(validation["errors"]) == 0
            assert len(validation["warnings"]) == 0
    
    def test_validate_embedding_result_missing_chunk_id(self, valid_config, valid_embedding_result):
        """Test validation with missing chunk_id."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            valid_embedding_result.chunk_id = ""
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is False
            assert "Missing chunk_id" in validation["errors"]
    
    def test_validate_embedding_result_missing_embedding(self, valid_config, valid_embedding_result):
        """Test validation with missing embedding."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            valid_embedding_result.embedding = []
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is False
            assert "Missing embedding" in validation["errors"]
    
    def test_validate_embedding_result_invalid_dimensions(self, valid_config, valid_embedding_result):
        """Test validation with invalid embedding dimensions."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            valid_embedding_result.embedding = [0.1] * 512  # Wrong dimension
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is False
            assert "Invalid embedding dimensions" in validation["errors"][0]
    
    def test_validate_embedding_result_invalid_values(self, valid_config, valid_embedding_result):
        """Test validation with invalid embedding values."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            valid_embedding_result.embedding = [0.1, float('nan')] + [0.1] * 1534
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is False
            assert "Invalid embedding values" in validation["errors"][0]
    
    def test_validate_embedding_result_dimension_mismatch(self, valid_config, valid_embedding_result):
        """Test validation with dimension mismatch in metadata."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            valid_embedding_result.embedding_metadata.embedding_dimension = 512
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is False
            assert "Embedding dimension mismatch with metadata" in validation["errors"]
    
    def test_validate_embedding_result_invalid_processing_time(self, valid_config, valid_embedding_result):
        """Test validation with invalid processing time."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            valid_embedding_result.embedding_metadata.processing_time = -1.0
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is False
            assert "Invalid processing time" in validation["errors"]
    
    def test_validate_embedding_result_warnings(self, valid_config, valid_embedding_result):
        """Test validation warnings."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(valid_config)
            
            valid_embedding_result.model = ""
            valid_embedding_result.created_at = None
            validation = client.validate_embedding_result(valid_embedding_result)
            
            assert validation["valid"] is True  # Still valid, just warnings
            assert "Missing model information" in validation["warnings"]
            assert "Missing creation timestamp" in validation["warnings"]
    
    def test_embedding_validation_in_generate_embedding(self, valid_config):
        """Test that validation is called during embedding generation."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock response with invalid embedding (wrong dimension)
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 512)]  # Wrong dimension
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(valid_config)
            
            with pytest.raises(ValueError, match="Invalid embedding result"):
                client.generate_embedding("Hello world", "chunk_1")
    
    def test_embedding_validation_in_batch_processing(self, valid_config):
        """Test that validation is called during batch processing."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock response with invalid embedding (wrong dimension)
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 512)]  # Wrong dimension
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(valid_config)
            
            requests = [EmbeddingRequest("Hello", "chunk_1")]
            
            with pytest.raises(ValueError, match="Invalid embedding result"):
                client.generate_embeddings_batch(requests)
    
    def test_mock_client_validation(self, valid_config):
        """Test that mock client produces valid embeddings."""
        mock_client = MockEmbeddingClient(valid_config)
        
        result = mock_client.generate_embedding("Hello world", "chunk_1")
        
        # Test that mock client produces valid results
        assert mock_client.validate_embedding(result.embedding)
        assert mock_client.validate_embedding_dimensions(result.embedding)
        assert mock_client.validate_embedding_values(result.embedding)
        
        validation = mock_client.validate_embedding_result(result)
        assert validation["valid"] is True


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def rate_limit_config(self):
        """Create config with tight rate limits for testing."""
        return OpenAIConfig(
            api_key="test_api_key",
            model="text-embedding-ada-002",
            rate_limit_requests=5,
            rate_limit_window=10
        )
    
    def test_rate_limiter_basic_functionality(self):
        """Test basic rate limiter functionality."""
        rate_limiter = RateLimiter(max_requests=5, time_window=10)
        
        # Should be able to acquire 5 tokens
        for _ in range(5):
            assert rate_limiter.acquire(tokens=1) is True
        
        # 6th request should fail
        assert rate_limiter.acquire(tokens=1) is False
    
    def test_rate_limiter_token_refill(self):
        """Test that tokens are refilled over time."""
        rate_limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Use up all tokens
        assert rate_limiter.acquire(tokens=1) is True
        assert rate_limiter.acquire(tokens=1) is True
        assert rate_limiter.acquire(tokens=1) is False
        
        # Wait for tokens to refill
        time.sleep(1.1)
        
        # Should be able to acquire tokens again
        assert rate_limiter.acquire(tokens=1) is True
    
    def test_rate_limiter_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        rate_limiter = RateLimiter(max_requests=10, time_window=60)
        
        # Acquire 5 tokens at once
        assert rate_limiter.acquire(tokens=5) is True
        
        # Should have 5 tokens left
        assert rate_limiter.acquire(tokens=5) is True
        
        # Should fail to acquire 1 more token
        assert rate_limiter.acquire(tokens=1) is False
    
    def test_rate_limiter_wait_for_availability(self):
        """Test waiting for token availability."""
        rate_limiter = RateLimiter(max_requests=1, time_window=1)
        
        # Use up the token
        assert rate_limiter.acquire(tokens=1) is True
        
        # Wait for availability should work
        start_time = time.time()
        wait_time = rate_limiter.wait_for_availability(tokens=1)
        elapsed = time.time() - start_time
        
        assert wait_time > 0
        assert elapsed >= 1.0  # Should wait at least 1 second
    
    def test_rate_limiter_status(self):
        """Test rate limiter status reporting."""
        rate_limiter = RateLimiter(max_requests=5, time_window=10)
        
        # Initial status
        status = rate_limiter.get_status()
        assert status["current_requests"] == 0
        assert status["max_requests"] == 5
        assert status["time_window"] == 10
        assert status["requests_remaining"] == 5
        
        # After acquiring some tokens
        rate_limiter.acquire(tokens=3)
        status = rate_limiter.get_status()
        assert status["current_requests"] == 3
        assert status["requests_remaining"] == 2
    
    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        rate_limiter = RateLimiter(max_requests=10, time_window=60)
        successful_acquisitions = []
        
        def acquire_tokens():
            for _ in range(5):
                if rate_limiter.acquire(tokens=1):
                    successful_acquisitions.append(1)
                time.sleep(0.01)  # Small delay to increase contention
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=acquire_tokens)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not exceed the limit
        assert len(successful_acquisitions) <= 10
    
    def test_embedding_client_rate_limiting(self, rate_limit_config):
        """Test that embedding client applies rate limiting."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(rate_limit_config)
            
            # Make requests up to the limit
            for i in range(5):
                result = client.generate_embedding(f"Text {i}", f"chunk_{i}")
                assert result.chunk_id == f"chunk_{i}"
            
            # Check rate limiter status
            status = client.get_rate_limit_status()
            assert status["current_requests"] == 5
            assert status["requests_remaining"] == 0
    
    def test_embedding_client_rate_limiting_with_wait(self, rate_limit_config):
        """Test rate limiting with waiting."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(rate_limit_config)
            
            # Use up the rate limit
            for i in range(5):
                client.generate_embedding(f"Text {i}", f"chunk_{i}")
            
            # This should wait for rate limit to reset
            start_time = time.time()
            result = client.generate_embedding("Text 6", "chunk_6")
            elapsed = time.time() - start_time
            
            assert result.chunk_id == "chunk_6"
            assert elapsed > 0  # Should have waited
    
    def test_rate_limiter_reset(self, rate_limit_config):
        """Test rate limiter reset functionality."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(rate_limit_config)
            
            # Use up the rate limit
            for i in range(5):
                client.generate_embedding(f"Text {i}", f"chunk_{i}")
            
            # Check that rate limit is exhausted
            status = client.get_rate_limit_status()
            assert status["requests_remaining"] == 0
            
            # Reset the rate limiter
            client.reset_rate_limiter()
            
            # Should be able to make requests again
            status = client.get_rate_limit_status()
            assert status["requests_remaining"] == 5
    
    def test_batch_processing_with_rate_limiting(self, rate_limit_config):
        """Test batch processing with rate limiting."""
        with patch('src.embedder.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1] * 1536),
                Mock(embedding=[0.2] * 1536)
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            client = EmbeddingClient(rate_limit_config)
            
            # Create requests
            requests = [
                EmbeddingRequest(f"Text {i}", f"chunk_{i}")
                for i in range(10)
            ]
            
            # Process in batches - should apply rate limiting
            results = client.generate_embeddings_batch(requests)
            
            assert len(results) == 10
            
            # Check that rate limiting was applied
            status = client.get_rate_limit_status()
            assert status["current_requests"] > 0
    
    def test_model_info_includes_rate_limiting(self, rate_limit_config):
        """Test that model info includes rate limiting configuration."""
        with patch('src.embedder.OpenAI'):
            client = EmbeddingClient(rate_limit_config)
            
            info = client.get_model_info()
            
            assert "rate_limit_requests" in info
            assert "rate_limit_window" in info
            assert info["rate_limit_requests"] == 5
            assert info["rate_limit_window"] == 10
    
    def test_mock_client_rate_limiting(self, rate_limit_config):
        """Test that mock client also supports rate limiting."""
        mock_client = MockEmbeddingClient(rate_limit_config)
        
        # Should be able to check rate limit status
        status = mock_client.get_rate_limit_status()
        assert status["max_requests"] == 5
        assert status["time_window"] == 10
        
        # Should be able to reset rate limiter
        mock_client.reset_rate_limiter()
        assert mock_client.get_rate_limit_status()["requests_remaining"] == 5