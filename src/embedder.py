"""OpenAI embedding integration module."""

import os
import time
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import deque

from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel, Field

from .config import OpenAIConfig


logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls using token bucket algorithm."""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens for API call.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired successfully, False otherwise
        """
        with self.lock:
            current_time = time.time()
            
            # Remove old requests outside the time window
            while self.requests and self.requests[0] <= current_time - self.time_window:
                self.requests.popleft()
            
            # Check if we can make the request
            if len(self.requests) + tokens <= self.max_requests:
                # Add tokens to the bucket
                for _ in range(tokens):
                    self.requests.append(current_time)
                return True
            
            return False
    
    def wait_for_availability(self, tokens: int = 1) -> float:
        """Wait for tokens to become available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time waited in seconds
        """
        start_time = time.time()
        
        while not self.acquire(tokens):
            time.sleep(0.1)  # Small sleep to prevent busy waiting
        
        return time.time() - start_time
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status.
        
        Returns:
            Dictionary with rate limiter status
        """
        with self.lock:
            current_time = time.time()
            
            # Clean up old requests
            while self.requests and self.requests[0] <= current_time - self.time_window:
                self.requests.popleft()
            
            return {
                "current_requests": len(self.requests),
                "max_requests": self.max_requests,
                "time_window": self.time_window,
                "requests_remaining": self.max_requests - len(self.requests),
                "oldest_request_age": current_time - self.requests[0] if self.requests else 0
            }


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    text: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    chunk_id: str
    embedding: List[float]
    model: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    embedding_metadata: Optional["EmbeddingMetadata"] = None


class EmbeddingMetadata(BaseModel):
    """Metadata for embedding tracking."""
    model: str = Field(..., description="OpenAI model used")
    model_version: str = Field(..., description="Model version")
    embedding_version: str = Field(default="1.0", description="Embedding schema version")
    created_at: datetime = Field(..., description="Creation timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    processing_time: float = Field(..., description="Processing time in seconds")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    batch_size: int = Field(default=1, description="Batch size used for generation")
    retry_count: int = Field(default=0, description="Number of retries performed")
    openai_api_version: str = Field(default="v1", description="OpenAI API version used")
    client_version: str = Field(default="1.0", description="Client library version")
    input_tokens: Optional[int] = Field(None, description="Number of input tokens")
    fingerprint: Optional[str] = Field(None, description="Model fingerprint for versioning")


class EmbeddingClient:
    """OpenAI embedding client with retry logic and batching."""
    
    def __init__(self, config: OpenAIConfig):
        """Initialize the embedding client.
        
        Args:
            config: OpenAI configuration
        """
        self.config = config
        self.client = None
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            time_window=config.rate_limit_window
        )
        self._setup_client()
    
    def _setup_client(self):
        """Set up the OpenAI client with API key from config."""
        if not self.config.api_key or self.config.api_key.startswith("your_"):
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
        
        logger.info(f"OpenAI client initialized with model: {self.config.model}")
    
    def generate_embedding(self, text: str, chunk_id: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """Generate embedding for a single text chunk.
        
        Args:
            text: Text to embed
            chunk_id: Unique identifier for the chunk
            metadata: Optional metadata to include
            
        Returns:
            EmbeddingResult with generated embedding
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        start_time = time.time()
        retry_count = 0
        
        try:
            response, retry_count = self._call_openai_api_with_retry_tracking([text])
            embedding = response.data[0].embedding
            
            processing_time = time.time() - start_time
            
            # Create comprehensive metadata
            embedding_metadata = self._create_embedding_metadata(
                text=text,
                processing_time=processing_time,
                retry_count=retry_count,
                batch_size=1
            )
            
            result = EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                model=self.config.model,
                created_at=datetime.now(),
                metadata=metadata or {},
                version="1.0",
                embedding_metadata=embedding_metadata
            )
            
            # Validate the result before returning
            validation = self.validate_embedding_result(result)
            if not validation["valid"]:
                error_msg = f"Invalid embedding result: {', '.join(validation['errors'])}"
                logger.error(f"Embedding validation failed for chunk {chunk_id}: {error_msg}")
                raise ValueError(error_msg)
            
            if validation["warnings"]:
                logger.warning(f"Embedding validation warnings for chunk {chunk_id}: {', '.join(validation['warnings'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for chunk {chunk_id}: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of text chunks.
        
        Args:
            requests: List of embedding requests
            
        Returns:
            List of EmbeddingResult objects
        """
        if not requests:
            return []
        
        # Split into batches based on config
        batches = self._split_into_batches(requests)
        results = []
        
        for batch in batches:
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _split_into_batches(self, requests: List[EmbeddingRequest]) -> List[List[EmbeddingRequest]]:
        """Split requests into batches based on configured batch size."""
        batch_size = self.config.batch_size
        batches = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_batch(self, batch: List[EmbeddingRequest]) -> List[EmbeddingResult]:
        """Process a single batch of embedding requests."""
        texts = [req.text for req in batch]
        start_time = time.time()
        
        try:
            response, retry_count = self._call_openai_api_with_retry_tracking(texts)
            processing_time = time.time() - start_time
            
            results = []
            for i, req in enumerate(batch):
                embedding = response.data[i].embedding
                
                # Create metadata for this specific embedding
                embedding_metadata = self._create_embedding_metadata(
                    text=req.text,
                    processing_time=processing_time / len(batch),  # Approximate per-embedding time
                    retry_count=retry_count,
                    batch_size=len(batch)
                )
                
                result = EmbeddingResult(
                    chunk_id=req.chunk_id,
                    embedding=embedding,
                    model=self.config.model,
                    created_at=datetime.now(),
                    metadata=req.metadata,
                    version="1.0",
                    embedding_metadata=embedding_metadata
                )
                
                # Validate the result before adding to results
                validation = self.validate_embedding_result(result)
                if not validation["valid"]:
                    error_msg = f"Invalid embedding result: {', '.join(validation['errors'])}"
                    logger.error(f"Embedding validation failed for chunk {req.chunk_id}: {error_msg}")
                    raise ValueError(error_msg)
                
                if validation["warnings"]:
                    logger.warning(f"Embedding validation warnings for chunk {req.chunk_id}: {', '.join(validation['warnings'])}")
                
                results.append(result)
            
            logger.info(f"Processed batch of {len(batch)} embeddings in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process batch of {len(batch)} embeddings: {str(e)}")
            raise
    
    def _call_openai_api(self, texts: List[str]) -> CreateEmbeddingResponse:
        """Call OpenAI API with retry logic and exponential backoff."""
        response, _ = self._call_openai_api_with_retry_tracking(texts)
        return response
    
    def _call_openai_api_with_retry_tracking(self, texts: List[str]) -> tuple[CreateEmbeddingResponse, int]:
        """Call OpenAI API with retry logic and return retry count.
        
        Returns:
            Tuple of (response, retry_count)
        """
        # Apply rate limiting before making the API call
        wait_time = self.rate_limiter.wait_for_availability(tokens=1)
        if wait_time > 0:
            logger.info(f"Rate limit applied, waited {wait_time:.2f}s before API call")
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts
                )
                return response, attempt
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Max retries exceeded for OpenAI API call: {str(e)}")
                    raise
                
                # Exponential backoff
                wait_time = (2 ** attempt) * 1  # 1s, 2s, 4s, 8s...
                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        # This should never be reached
        raise RuntimeError("Unexpected error in retry logic")
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding is properly formatted.
        
        Args:
            embedding: List of floats representing the embedding
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(embedding, list):
            return False
        
        if len(embedding) == 0:
            return False
        
        # Check that all values are floats
        for value in embedding:
            if not isinstance(value, (int, float)):
                return False
        
        return True
    
    def validate_embedding_dimensions(self, embedding: List[float], expected_dim: int = 1536) -> bool:
        """Validate that an embedding has the expected dimensions.
        
        Args:
            embedding: List of floats representing the embedding
            expected_dim: Expected dimension count (default 1536 for OpenAI ada-002)
            
        Returns:
            True if valid dimensions, False otherwise
        """
        if not self.validate_embedding(embedding):
            return False
        
        return len(embedding) == expected_dim
    
    def validate_embedding_values(self, embedding: List[float]) -> bool:
        """Validate that embedding values are within expected ranges.
        
        Args:
            embedding: List of floats representing the embedding
            
        Returns:
            True if values are valid, False otherwise
        """
        if not self.validate_embedding(embedding):
            return False
        
        # Check for NaN, infinity, and extreme values
        for value in embedding:
            if value != value:  # NaN check
                return False
            if value == float('inf') or value == float('-inf'):
                return False
            if abs(value) > 100:  # Extreme values check
                return False
        
        return True
    
    def validate_embedding_result(self, result: EmbeddingResult) -> Dict[str, Any]:
        """Comprehensive validation of an embedding result.
        
        Args:
            result: EmbeddingResult to validate
            
        Returns:
            Dict with validation results and error messages
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate basic structure
        if not result.chunk_id:
            validation_result["valid"] = False
            validation_result["errors"].append("Missing chunk_id")
        
        if not result.embedding:
            validation_result["valid"] = False
            validation_result["errors"].append("Missing embedding")
        else:
            # Validate embedding format
            if not self.validate_embedding(result.embedding):
                validation_result["valid"] = False
                validation_result["errors"].append("Invalid embedding format")
            
            # Validate dimensions
            if not self.validate_embedding_dimensions(result.embedding):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid embedding dimensions: {len(result.embedding)}, expected 1536")
            
            # Validate values
            if not self.validate_embedding_values(result.embedding):
                validation_result["valid"] = False
                validation_result["errors"].append("Invalid embedding values (NaN, infinity, or extreme values)")
        
        # Validate metadata
        if not result.model:
            validation_result["warnings"].append("Missing model information")
        
        if not result.created_at:
            validation_result["warnings"].append("Missing creation timestamp")
        
        if result.embedding_metadata:
            # Validate embedding metadata
            if result.embedding_metadata.embedding_dimension != len(result.embedding):
                validation_result["valid"] = False
                validation_result["errors"].append("Embedding dimension mismatch with metadata")
            
            if result.embedding_metadata.processing_time < 0:
                validation_result["valid"] = False
                validation_result["errors"].append("Invalid processing time")
        
        return validation_result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.config.model,
            "batch_size": self.config.batch_size,
            "max_retries": self.config.max_retries,
            "timeout": self.config.timeout,
            "rate_limit_requests": self.config.rate_limit_requests,
            "rate_limit_window": self.config.rate_limit_window
        }
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status.
        
        Returns:
            Dictionary with rate limit status
        """
        return self.rate_limiter.get_status()
    
    def reset_rate_limiter(self):
        """Reset the rate limiter (useful for testing)."""
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit_requests,
            time_window=self.config.rate_limit_window
        )
    
    def _create_embedding_metadata(self, text: str, processing_time: float, retry_count: int = 0, 
                                   batch_size: int = 1, request_id: Optional[str] = None) -> EmbeddingMetadata:
        """Create embedding metadata for tracking.
        
        Args:
            text: Input text
            processing_time: Time taken to process
            retry_count: Number of retries performed
            batch_size: Batch size used
            request_id: Optional request ID
            
        Returns:
            EmbeddingMetadata object
        """
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4
        
        return EmbeddingMetadata(
            model=self.config.model,
            model_version="002",  # For text-embedding-ada-002
            embedding_version="1.0",
            created_at=datetime.now(),
            request_id=request_id,
            processing_time=processing_time,
            embedding_dimension=1536,  # OpenAI ada-002 embedding dimension
            batch_size=batch_size,
            retry_count=retry_count,
            openai_api_version="v1",
            client_version="1.0",
            input_tokens=estimated_tokens,
            fingerprint=f"{self.config.model}-{datetime.now().strftime('%Y%m%d')}"
        )


class MockEmbeddingClient(EmbeddingClient):
    """Mock embedding client for testing without API calls."""
    
    def __init__(self, config: OpenAIConfig):
        """Initialize mock client."""
        self.config = config
        self.client = None  # No actual OpenAI client
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            time_window=config.rate_limit_window
        )
        logger.info("Mock embedding client initialized")
    
    def _setup_client(self):
        """Mock client setup - no actual API client."""
        pass
    
    def _call_openai_api(self, texts: List[str]) -> CreateEmbeddingResponse:
        """Mock OpenAI API call that returns deterministic embeddings."""
        response, _ = self._call_openai_api_with_retry_tracking(texts)
        return response
    
    def _call_openai_api_with_retry_tracking(self, texts: List[str]) -> tuple[CreateEmbeddingResponse, int]:
        """Mock OpenAI API call that returns deterministic embeddings with retry tracking."""
        # Generate deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on text hash
            embedding = [float(hash(text + str(i)) % 1000) / 1000.0 for i in range(1536)]
            embeddings.append(embedding)
        
        # Create mock response object
        class MockEmbeddingData:
            def __init__(self, embedding):
                self.embedding = embedding
        
        class MockResponse:
            def __init__(self, embeddings):
                self.data = [MockEmbeddingData(emb) for emb in embeddings]
        
        return MockResponse(embeddings), 0  # Mock always succeeds on first try


def create_embedding_client(config: OpenAIConfig, use_mock: bool = False) -> EmbeddingClient:
    """Factory function to create embedding client.
    
    Args:
        config: OpenAI configuration
        use_mock: Whether to use mock client for testing
        
    Returns:
        EmbeddingClient instance
    """
    if use_mock:
        return MockEmbeddingClient(config)
    else:
        return EmbeddingClient(config)