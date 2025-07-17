"""
Tests for the chunker module, focusing on chunk ID generation and idempotency.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime
from typing import List, Dict, Any

from src.chunker import (
    Chunk, ChunkMetadata, ChunkIDGenerator, BaseChunker, FixedWindowChunker, SpeakerBasedChunker,
    create_chunker_from_config, get_default_chunking_config
)


class TestChunkIDGenerator:
    """Test suite for ChunkIDGenerator class."""
    
    def test_generate_chunk_id_deterministic(self):
        """Test that chunk ID generation is deterministic."""
        content = "Hello, this is a test transcript content."
        source_file = "test_file.json"
        start_time = 10.5
        end_time = 25.3
        chunk_type = "fixed_window"
        
        # Generate the same ID multiple times
        id1 = ChunkIDGenerator.generate_chunk_id(
            content, source_file, start_time, end_time, chunk_type
        )
        id2 = ChunkIDGenerator.generate_chunk_id(
            content, source_file, start_time, end_time, chunk_type
        )
        id3 = ChunkIDGenerator.generate_chunk_id(
            content, source_file, start_time, end_time, chunk_type
        )
        
        assert id1 == id2 == id3
        assert id1.startswith("chunk_")
        assert len(id1) == 22  # "chunk_" + 16 hex characters
    
    def test_generate_chunk_id_unique_for_different_content(self):
        """Test that different content generates different IDs."""
        base_params = {
            "source_file": "test_file.json",
            "start_time": 10.5,
            "end_time": 25.3,
            "chunk_type": "fixed_window"
        }
        
        id1 = ChunkIDGenerator.generate_chunk_id(
            content="First content", **base_params
        )
        id2 = ChunkIDGenerator.generate_chunk_id(
            content="Second content", **base_params
        )
        
        assert id1 != id2
        assert both_start_with_chunk_prefix(id1, id2)
    
    def test_generate_chunk_id_unique_for_different_timestamps(self):
        """Test that different timestamps generate different IDs."""
        base_params = {
            "content": "Same content",
            "source_file": "test_file.json",
            "chunk_type": "fixed_window"
        }
        
        id1 = ChunkIDGenerator.generate_chunk_id(
            start_time=10.5, end_time=25.3, **base_params
        )
        id2 = ChunkIDGenerator.generate_chunk_id(
            start_time=15.0, end_time=30.0, **base_params
        )
        
        assert id1 != id2
        assert both_start_with_chunk_prefix(id1, id2)
    
    def test_generate_chunk_id_unique_for_different_source_files(self):
        """Test that different source files generate different IDs."""
        base_params = {
            "content": "Same content",
            "start_time": 10.5,
            "end_time": 25.3,
            "chunk_type": "fixed_window"
        }
        
        id1 = ChunkIDGenerator.generate_chunk_id(
            source_file="file1.json", **base_params
        )
        id2 = ChunkIDGenerator.generate_chunk_id(
            source_file="file2.json", **base_params
        )
        
        assert id1 != id2
        assert both_start_with_chunk_prefix(id1, id2)
    
    def test_generate_chunk_id_with_additional_context(self):
        """Test chunk ID generation with additional context."""
        base_params = {
            "content": "Test content",
            "source_file": "test_file.json",
            "start_time": 10.5,
            "end_time": 25.3,
            "chunk_type": "fixed_window"
        }
        
        id1 = ChunkIDGenerator.generate_chunk_id(**base_params)
        id2 = ChunkIDGenerator.generate_chunk_id(
            additional_context={"speaker": "John"}, **base_params
        )
        
        assert id1 != id2
        assert both_start_with_chunk_prefix(id1, id2)
    
    def test_normalize_content_whitespace(self):
        """Test content normalization handles whitespace correctly."""
        content1 = "  Hello   world  \n\t  "
        content2 = "Hello world"
        
        normalized1 = ChunkIDGenerator._normalize_content(content1)
        normalized2 = ChunkIDGenerator._normalize_content(content2)
        
        assert normalized1 == normalized2
        assert normalized1 == "hello world"
    
    def test_normalize_content_case_insensitive(self):
        """Test content normalization is case insensitive."""
        content1 = "Hello World"
        content2 = "hello world"
        content3 = "HELLO WORLD"
        
        normalized1 = ChunkIDGenerator._normalize_content(content1)
        normalized2 = ChunkIDGenerator._normalize_content(content2)
        normalized3 = ChunkIDGenerator._normalize_content(content3)
        
        assert normalized1 == normalized2 == normalized3
        assert normalized1 == "hello world"
    
    def test_normalize_content_punctuation(self):
        """Test content normalization handles punctuation."""
        content1 = 'Hello "world" test'
        content2 = "Hello world test"
        
        normalized1 = ChunkIDGenerator._normalize_content(content1)
        normalized2 = ChunkIDGenerator._normalize_content(content2)
        
        assert normalized1 == normalized2
        assert normalized1 == "hello world test"
    
    def test_detect_collision_no_collision(self):
        """Test collision detection when no collision exists."""
        chunk_id = "chunk_1234567890abcdef"
        existing_chunks = [
            create_mock_chunk("chunk_abcdef1234567890"),
            create_mock_chunk("chunk_fedcba0987654321")
        ]
        
        collision = ChunkIDGenerator.detect_collision(chunk_id, existing_chunks)
        assert collision is False
    
    def test_detect_collision_with_collision(self):
        """Test collision detection when collision exists."""
        chunk_id = "chunk_1234567890abcdef"
        existing_chunks = [
            create_mock_chunk("chunk_abcdef1234567890"),
            create_mock_chunk("chunk_1234567890abcdef"),  # Collision
            create_mock_chunk("chunk_fedcba0987654321")
        ]
        
        collision = ChunkIDGenerator.detect_collision(chunk_id, existing_chunks)
        assert collision is True
    
    def test_handle_collision_generates_different_id(self):
        """Test collision handling generates different ID."""
        base_params = {
            "content": "Test content",
            "source_file": "test_file.json",
            "start_time": 10.5,
            "end_time": 25.3,
            "chunk_type": "fixed_window"
        }
        
        original_id = ChunkIDGenerator.generate_chunk_id(**base_params)
        collision_id = ChunkIDGenerator.handle_collision(**base_params)
        
        assert original_id != collision_id
        assert both_start_with_chunk_prefix(original_id, collision_id)
    
    def test_handle_collision_with_counter(self):
        """Test collision handling with different counters."""
        base_params = {
            "content": "Test content",
            "source_file": "test_file.json",
            "start_time": 10.5,
            "end_time": 25.3,
            "chunk_type": "fixed_window"
        }
        
        id1 = ChunkIDGenerator.handle_collision(collision_counter=1, **base_params)
        id2 = ChunkIDGenerator.handle_collision(collision_counter=2, **base_params)
        
        assert id1 != id2
        assert both_start_with_chunk_prefix(id1, id2)
    
    def test_timestamp_rounding_consistency(self):
        """Test that timestamp rounding ensures consistency."""
        base_params = {
            "content": "Test content",
            "source_file": "test_file.json",
            "chunk_type": "fixed_window"
        }
        
        # These should generate the same ID due to rounding
        id1 = ChunkIDGenerator.generate_chunk_id(
            start_time=10.5001, end_time=25.3001, **base_params
        )
        id2 = ChunkIDGenerator.generate_chunk_id(
            start_time=10.5002, end_time=25.3002, **base_params
        )
        
        assert id1 == id2
        
        # These should generate different IDs
        id3 = ChunkIDGenerator.generate_chunk_id(
            start_time=10.501, end_time=25.301, **base_params
        )
        
        assert id1 != id3


class TestChunkMetadata:
    """Test suite for ChunkMetadata class."""
    
    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata creation with all fields."""
        metadata = ChunkMetadata(
            chunk_id="chunk_1234567890abcdef",
            source_file="test_file.json",
            start_time=10.5,
            end_time=25.3,
            speaker_count=2,
            speakers=["Alice", "Bob"],
            word_count=50,
            chunk_type="fixed_window",
            chunk_index=1,
            total_chunks=10
        )
        
        assert metadata.chunk_id == "chunk_1234567890abcdef"
        assert metadata.source_file == "test_file.json"
        assert metadata.start_time == 10.5
        assert metadata.end_time == 25.3
        assert metadata.speaker_count == 2
        assert metadata.speakers == ["Alice", "Bob"]
        assert metadata.word_count == 50
        assert metadata.chunk_type == "fixed_window"
        assert metadata.chunk_index == 1
        assert metadata.total_chunks == 10
        assert isinstance(metadata.created_at, datetime)
    
    def test_chunk_metadata_to_dict(self):
        """Test ChunkMetadata serialization to dictionary."""
        metadata = ChunkMetadata(
            chunk_id="chunk_1234567890abcdef",
            source_file="test_file.json",
            start_time=10.5,
            end_time=25.3,
            speaker_count=2,
            speakers=["Alice", "Bob"],
            word_count=50,
            chunk_type="fixed_window",
            chunk_index=1,
            total_chunks=10
        )
        
        result = metadata.to_dict()
        
        assert result["chunk_id"] == "chunk_1234567890abcdef"
        assert result["source_file"] == "test_file.json"
        assert result["start_time"] == 10.5
        assert result["end_time"] == 25.3
        assert result["speaker_count"] == 2
        assert result["speakers"] == ["Alice", "Bob"]
        assert result["word_count"] == 50
        assert result["chunk_type"] == "fixed_window"
        assert result["chunk_index"] == 1
        assert result["total_chunks"] == 10
        assert "created_at" in result
        assert isinstance(result["created_at"], str)


class TestChunk:
    """Test suite for Chunk class."""
    
    def test_chunk_creation(self):
        """Test Chunk creation with content and metadata."""
        metadata = ChunkMetadata(
            chunk_id="chunk_1234567890abcdef",
            source_file="test_file.json",
            start_time=10.5,
            end_time=25.3,
            speaker_count=1,
            speakers=["Alice"],
            word_count=10,
            chunk_type="fixed_window"
        )
        
        chunk = Chunk(
            content="Hello, this is test content.",
            metadata=metadata
        )
        
        assert chunk.content == "Hello, this is test content."
        assert chunk.metadata == metadata
    
    def test_chunk_to_dict(self):
        """Test Chunk serialization to dictionary."""
        metadata = ChunkMetadata(
            chunk_id="chunk_1234567890abcdef",
            source_file="test_file.json",
            start_time=10.5,
            end_time=25.3,
            speaker_count=1,
            speakers=["Alice"],
            word_count=10,
            chunk_type="fixed_window"
        )
        
        chunk = Chunk(
            content="Hello, this is test content.",
            metadata=metadata
        )
        
        result = chunk.to_dict()
        
        assert result["content"] == "Hello, this is test content."
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)
        assert result["metadata"]["chunk_id"] == "chunk_1234567890abcdef"


class TestBaseChunker:
    """Test suite for BaseChunker abstract class."""
    
    def test_base_chunker_initialization(self):
        """Test BaseChunker initialization with config."""
        config = {"window_size": 100, "overlap": 20}
        
        # Create a concrete implementation for testing
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker(config)
        
        assert chunker.config == config
        assert chunker._chunk_id_generator is not None
        assert isinstance(chunker._chunk_id_generator, ChunkIDGenerator)
    
    def test_base_chunker_create_chunk_metadata(self):
        """Test BaseChunker chunk metadata creation."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        metadata = chunker._create_chunk_metadata(
            content="Test content",
            source_file="test_file.json",
            start_time=10.5,
            end_time=25.3,
            speakers=["Alice", "Bob"],
            chunk_type="test_chunk",
            chunk_index=1,
            total_chunks=5
        )
        
        assert metadata.chunk_id.startswith("chunk_")
        assert metadata.source_file == "test_file.json"
        assert metadata.start_time == 10.5
        assert metadata.end_time == 25.3
        assert metadata.speaker_count == 2
        assert set(metadata.speakers) == {"Alice", "Bob"}
        assert metadata.word_count == 2  # "Test content"
        assert metadata.chunk_type == "test_chunk"
        assert metadata.chunk_index == 1
        assert metadata.total_chunks == 5
    
    def test_base_chunker_validate_chunk_uniqueness_valid(self):
        """Test chunk uniqueness validation with unique chunks."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunks = [
            create_mock_chunk("chunk_1234567890abcdef"),
            create_mock_chunk("chunk_abcdef1234567890"),
            create_mock_chunk("chunk_fedcba0987654321")
        ]
        
        assert chunker._validate_chunk_uniqueness(chunks) is True
    
    def test_base_chunker_validate_chunk_uniqueness_invalid(self):
        """Test chunk uniqueness validation with duplicate chunks."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunks = [
            create_mock_chunk("chunk_1234567890abcdef"),
            create_mock_chunk("chunk_abcdef1234567890"),
            create_mock_chunk("chunk_1234567890abcdef")  # Duplicate
        ]
        
        assert chunker._validate_chunk_uniqueness(chunks) is False
    
    def test_base_chunker_abstract_method(self):
        """Test that BaseChunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChunker()


class TestChunkIDIntegration:
    """Integration tests for chunk ID generation system."""
    
    def test_end_to_end_chunk_id_generation(self):
        """Test complete chunk ID generation workflow."""
        # Simulate creating chunks with the same content but different contexts
        content = "Hello, this is a test transcript."
        source_file = "meeting_001.json"
        start_time = 120.5
        end_time = 135.8
        chunk_type = "speaker_based"
        
        # Generate chunk ID
        chunk_id = ChunkIDGenerator.generate_chunk_id(
            content=content,
            source_file=source_file,
            start_time=start_time,
            end_time=end_time,
            chunk_type=chunk_type
        )
        
        # Verify ID format
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 22
        
        # Verify reproducibility
        chunk_id_2 = ChunkIDGenerator.generate_chunk_id(
            content=content,
            source_file=source_file,
            start_time=start_time,
            end_time=end_time,
            chunk_type=chunk_type
        )
        
        assert chunk_id == chunk_id_2
    
    def test_collision_handling_workflow(self):
        """Test complete collision handling workflow."""
        existing_chunks = []
        
        # Create first chunk
        chunk_id_1 = ChunkIDGenerator.generate_chunk_id(
            content="Test content",
            source_file="test.json",
            start_time=10.0,
            end_time=20.0,
            chunk_type="fixed_window"
        )
        
        existing_chunks.append(create_mock_chunk(chunk_id_1))
        
        # Try to create a "colliding" chunk (same ID)
        collision_detected = ChunkIDGenerator.detect_collision(
            chunk_id_1, existing_chunks
        )
        
        assert collision_detected is True
        
        # Handle collision
        new_chunk_id = ChunkIDGenerator.handle_collision(
            content="Test content",
            source_file="test.json",
            start_time=10.0,
            end_time=20.0,
            chunk_type="fixed_window"
        )
        
        assert new_chunk_id != chunk_id_1
        assert new_chunk_id.startswith("chunk_")
        
        # Verify no collision with new ID
        collision_detected_2 = ChunkIDGenerator.detect_collision(
            new_chunk_id, existing_chunks
        )
        
        assert collision_detected_2 is False
    
    def test_large_content_handling(self):
        """Test chunk ID generation with large content."""
        large_content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 1000
        
        chunk_id = ChunkIDGenerator.generate_chunk_id(
            content=large_content,
            source_file="large_file.json",
            start_time=0.0,
            end_time=600.0,
            chunk_type="fixed_window"
        )
        
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 22
        
        # Verify deterministic with large content
        chunk_id_2 = ChunkIDGenerator.generate_chunk_id(
            content=large_content,
            source_file="large_file.json",
            start_time=0.0,
            end_time=600.0,
            chunk_type="fixed_window"
        )
        
        assert chunk_id == chunk_id_2


# Helper functions for tests
def both_start_with_chunk_prefix(id1: str, id2: str) -> bool:
    """Check if both IDs start with chunk prefix."""
    return id1.startswith("chunk_") and id2.startswith("chunk_")


def create_mock_chunk(chunk_id: str) -> Chunk:
    """Create a mock chunk with given ID."""
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        source_file="test_file.json",
        start_time=0.0,
        end_time=10.0,
        speaker_count=1,
        speakers=["Test"],
        word_count=5,
        chunk_type="test"
    )
    
    return Chunk(
        content="Test content",
        metadata=metadata
    )


# Coverage and edge case tests
class TestDuplicateDetectionAndPrevention:
    """Test suite for duplicate detection and prevention logic."""
    
    def test_detect_duplicate_chunk_exact_match(self):
        """Test detection of exact duplicate chunks."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Create original chunk
        original_chunk = create_mock_chunk("chunk_1234567890abcdef")
        chunker._chunk_registry[original_chunk.metadata.chunk_id] = original_chunk
        
        # Create duplicate chunk with same ID and content
        duplicate_chunk = create_mock_chunk("chunk_1234567890abcdef")
        
        assert chunker._detect_duplicate_chunk(duplicate_chunk) is True
    
    def test_detect_duplicate_chunk_id_collision(self):
        """Test detection of ID collision with different content."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Create original chunk
        original_chunk = create_mock_chunk("chunk_1234567890abcdef")
        chunker._chunk_registry[original_chunk.metadata.chunk_id] = original_chunk
        
        # Create chunk with same ID but different content
        collision_chunk = create_mock_chunk("chunk_1234567890abcdef")
        collision_chunk.content = "Different content"
        
        assert chunker._detect_duplicate_chunk(collision_chunk) is False
    
    def test_detect_duplicate_chunk_no_duplicate(self):
        """Test detection when no duplicate exists."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Create chunk with unique ID
        unique_chunk = create_mock_chunk("chunk_1234567890abcdef")
        
        assert chunker._detect_duplicate_chunk(unique_chunk) is False
    
    def test_prevent_duplicate_chunks_removes_duplicates(self):
        """Test that duplicate chunks are removed from the list."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Create chunks with some duplicates
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        chunk3 = create_mock_chunk("chunk_1234567890abcdef")  # Duplicate of chunk1
        
        chunks = [chunk1, chunk2, chunk3]
        
        # Add chunk1 to registry to simulate it being processed first
        chunker._chunk_registry[chunk1.metadata.chunk_id] = chunk1
        
        unique_chunks = chunker._prevent_duplicate_chunks(chunks)
        
        # Should only have chunk2 (chunk1 already in registry, chunk3 is duplicate)
        assert len(unique_chunks) == 1
        assert unique_chunks[0].metadata.chunk_id == "chunk_abcdef1234567890"
    
    def test_prevent_duplicate_chunks_handles_collisions(self):
        """Test that ID collisions are handled with new IDs."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Create chunks that would have same ID but different content
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2.content = "Different content"
        
        chunks = [chunk1, chunk2]
        
        unique_chunks = chunker._prevent_duplicate_chunks(chunks)
        
        # Should have both chunks but with different IDs
        assert len(unique_chunks) == 2
        assert unique_chunks[0].metadata.chunk_id != unique_chunks[1].metadata.chunk_id
        assert unique_chunks[1].metadata.chunk_id.startswith("chunk_")
    
    def test_is_chunk_content_duplicate_identical_content(self):
        """Test content duplication detection with identical content."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk1.content = "This is test content"
        
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        chunk2.content = "This is test content"
        
        assert chunker._is_chunk_content_duplicate(chunk1, chunk2) is True
    
    def test_is_chunk_content_duplicate_normalized_content(self):
        """Test content duplication detection with normalized content."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk1.content = "  This is TEST content  "
        
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        chunk2.content = "this is test content"
        
        assert chunker._is_chunk_content_duplicate(chunk1, chunk2) is True
    
    def test_is_chunk_content_duplicate_different_content(self):
        """Test content duplication detection with different content."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk1.content = "This is test content"
        
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        chunk2.content = "This is different content"
        
        assert chunker._is_chunk_content_duplicate(chunk1, chunk2) is False
    
    def test_detect_semantic_duplicates_removes_duplicates(self):
        """Test semantic duplicate detection removes similar content."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Create chunks with similar content
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk1.content = "This is test content"
        
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        chunk2.content = "This is different content"
        
        chunk3 = create_mock_chunk("chunk_fedcba0987654321")
        chunk3.content = "  THIS IS TEST CONTENT  "  # Normalized duplicate
        
        chunks = [chunk1, chunk2, chunk3]
        
        unique_chunks = chunker._detect_semantic_duplicates(chunks)
        
        # Should remove chunk3 as it's a semantic duplicate of chunk1
        assert len(unique_chunks) == 2
        assert unique_chunks[0].metadata.chunk_id == "chunk_1234567890abcdef"
        assert unique_chunks[1].metadata.chunk_id == "chunk_abcdef1234567890"
    
    def test_validate_chunk_integrity_valid_chunks(self):
        """Test chunk integrity validation with valid chunks."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        
        chunks = [chunk1, chunk2]
        
        assert chunker._validate_chunk_integrity(chunks) is True
    
    def test_validate_chunk_integrity_duplicate_ids(self):
        """Test chunk integrity validation with duplicate IDs."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2 = create_mock_chunk("chunk_1234567890abcdef")  # Duplicate ID
        
        chunks = [chunk1, chunk2]
        
        assert chunker._validate_chunk_integrity(chunks) is False
    
    def test_validate_chunk_integrity_invalid_chunk_id_format(self):
        """Test chunk integrity validation with invalid chunk ID format."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunk = create_mock_chunk("invalid_chunk_id")
        chunks = [chunk]
        
        assert chunker._validate_chunk_integrity(chunks) is False
        
        # Test with wrong length
        chunk2 = create_mock_chunk("chunk_123")  # Too short
        chunks2 = [chunk2]
        
        assert chunker._validate_chunk_integrity(chunks2) is False
    
    def test_validate_chunk_integrity_invalid_timestamps(self):
        """Test chunk integrity validation with invalid timestamps."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        chunk = create_mock_chunk("chunk_1234567890abcdef")
        # Set invalid timestamps (start >= end)
        chunk.metadata.start_time = 30.0
        chunk.metadata.end_time = 20.0
        
        chunks = [chunk]
        
        assert chunker._validate_chunk_integrity(chunks) is False
    
    def test_validate_chunk_integrity_empty_chunks(self):
        """Test chunk integrity validation with empty chunks list."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        assert chunker._validate_chunk_integrity([]) is True
    
    def test_clear_chunk_registry(self):
        """Test clearing the chunk registry."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Add some chunks to registry
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        chunker._chunk_registry[chunk1.metadata.chunk_id] = chunk1
        chunker._chunk_registry[chunk2.metadata.chunk_id] = chunk2
        
        assert len(chunker._chunk_registry) == 2
        
        chunker._clear_chunk_registry()
        
        assert len(chunker._chunk_registry) == 0
    
    def test_get_chunk_registry_stats(self):
        """Test getting chunk registry statistics."""
        class TestChunker(BaseChunker):
            def chunk(self, transcript_data):
                return []
        
        chunker = TestChunker()
        
        # Add some chunks to registry
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk1.content = "Test content 1"
        chunk2 = create_mock_chunk("chunk_abcdef1234567890")
        chunk2.content = "Test content 2"
        
        chunker._chunk_registry[chunk1.metadata.chunk_id] = chunk1
        chunker._chunk_registry[chunk2.metadata.chunk_id] = chunk2
        
        stats = chunker._get_chunk_registry_stats()
        
        assert stats["total_chunks"] == 2
        assert len(stats["chunk_ids"]) == 2
        assert "chunk_1234567890abcdef" in stats["chunk_ids"]
        assert "chunk_abcdef1234567890" in stats["chunk_ids"]
        assert stats["memory_usage_bytes"] > 0
    
    def test_duplicate_prevention_integration_fixed_window(self):
        """Test duplicate prevention integration with FixedWindowChunker."""
        config = {"window_size": 30, "overlap_seconds": 10}
        chunker = FixedWindowChunker(config)
        
        # Create transcript with overlapping windows that might create duplicates
        transcript_data = {
            "source_file": "duplicate_test.json",
            "segments": [
                {"start": 0.0, "end": 20.0, "speaker": "Alice", "text": "This is a test segment with enough content."},
                {"start": 20.0, "end": 40.0, "speaker": "Bob", "text": "Another test segment with sufficient content."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Verify no duplicates
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # Verify all chunks pass integrity checks
        assert chunker._validate_chunk_integrity(chunks) is True
    
    def test_duplicate_prevention_integration_speaker_based(self):
        """Test duplicate prevention integration with SpeakerBasedChunker."""
        config = {"merge_consecutive_same_speaker": True}
        chunker = SpeakerBasedChunker(config)
        
        # Create transcript with potential duplicates
        transcript_data = {
            "source_file": "duplicate_test.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "This is a test segment with enough content."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Another test segment with sufficient content."},
                {"start": 60.0, "end": 90.0, "speaker": "Alice", "text": "Final test segment with adequate content."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Verify no duplicates
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # Verify all chunks pass integrity checks
        assert chunker._validate_chunk_integrity(chunks) is True
    
    def test_duplicate_prevention_with_identical_content(self):
        """Test duplicate prevention with identical content from different sources."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 0})
        
        # Create transcript with identical content in different segments
        transcript_data = {
            "source_file": "identical_content.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "This is identical content for testing."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "This is identical content for testing."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should still create chunk as they have different speakers/timestamps
        assert len(chunks) == 1
        assert chunker._validate_chunk_integrity(chunks) is True
    
    def test_collision_handling_in_duplicate_prevention(self):
        """Test collision handling during duplicate prevention."""
        chunker = FixedWindowChunker()
        
        # Create chunks that would have colliding IDs
        chunk1 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2 = create_mock_chunk("chunk_1234567890abcdef")
        chunk2.content = "Different content"
        
        chunks = [chunk1, chunk2]
        
        unique_chunks = chunker._prevent_duplicate_chunks(chunks)
        
        # Should handle collision by generating new ID for second chunk
        assert len(unique_chunks) == 2
        assert unique_chunks[0].metadata.chunk_id != unique_chunks[1].metadata.chunk_id
        assert all(chunk.metadata.chunk_id.startswith("chunk_") for chunk in unique_chunks)
    
    def test_duplicate_prevention_performance_with_many_chunks(self):
        """Test duplicate prevention performance with many chunks."""
        chunker = FixedWindowChunker({"window_size": 30, "overlap_seconds": 5})
        
        # Create a large transcript
        segments = []
        for i in range(100):
            segments.append({
                "start": i * 15.0,
                "end": (i + 1) * 15.0,
                "speaker": f"Speaker{i % 3}",
                "text": f"This is segment {i} with unique content for testing performance."
            })
        
        transcript_data = {
            "source_file": "performance_test.json",
            "segments": segments
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should process all chunks without duplicates
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        assert chunker._validate_chunk_integrity(chunks) is True
        assert len(chunks) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        chunk_id = ChunkIDGenerator.generate_chunk_id(
            content="",
            source_file="test.json",
            start_time=0.0,
            end_time=1.0,
            chunk_type="fixed_window"
        )
        
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 22
    
    def test_special_characters_in_content(self):
        """Test handling of special characters in content."""
        content = "Hello! @#$%^&*()_+{}[]|\\:;\"'<>?,./"
        
        chunk_id = ChunkIDGenerator.generate_chunk_id(
            content=content,
            source_file="test.json",
            start_time=0.0,
            end_time=1.0,
            chunk_type="fixed_window"
        )
        
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 22
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode content."""
        content = "Hello 世界 🌍 café naïve résumé"
        
        chunk_id = ChunkIDGenerator.generate_chunk_id(
            content=content,
            source_file="test.json",
            start_time=0.0,
            end_time=1.0,
            chunk_type="fixed_window"
        )
        
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 22
    
    def test_extreme_timestamp_values(self):
        """Test handling of extreme timestamp values."""
        chunk_id = ChunkIDGenerator.generate_chunk_id(
            content="Test content",
            source_file="test.json",
            start_time=0.0,
            end_time=999999.999,
            chunk_type="fixed_window"
        )
        
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 22
    
    def test_long_file_path_handling(self):
        """Test handling of very long file paths."""
        long_path = "/".join(["very_long_directory_name"] * 20) + "/file.json"
        
        chunk_id = ChunkIDGenerator.generate_chunk_id(
            content="Test content",
            source_file=long_path,
            start_time=0.0,
            end_time=1.0,
            chunk_type="fixed_window"
        )
        
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 22
    
    def test_multiple_collision_handling(self):
        """Test handling multiple collisions in sequence."""
        base_params = {
            "content": "Test content",
            "source_file": "test.json",
            "start_time": 0.0,
            "end_time": 1.0,
            "chunk_type": "fixed_window"
        }
        
        # Generate IDs with different collision counters
        ids = []
        for i in range(1, 6):
            chunk_id = ChunkIDGenerator.handle_collision(
                collision_counter=i, **base_params
            )
            ids.append(chunk_id)
        
        # All IDs should be different
        assert len(set(ids)) == 5
        
        # All should start with chunk prefix
        for chunk_id in ids:
            assert chunk_id.startswith("chunk_")
            assert len(chunk_id) == 22


class TestFixedWindowChunker:
    """Test suite for FixedWindowChunker class."""
    
    def test_fixed_window_chunker_initialization_default_config(self):
        """Test FixedWindowChunker initialization with default configuration."""
        chunker = FixedWindowChunker()
        
        assert chunker.window_size == 60
        assert chunker.overlap_seconds == 5
        assert chunker.min_chunk_length == 10
        assert chunker.config == {}
    
    def test_fixed_window_chunker_initialization_custom_config(self):
        """Test FixedWindowChunker initialization with custom configuration."""
        config = {
            "window_size": 120,
            "overlap_seconds": 10,
            "min_chunk_length": 20
        }
        
        chunker = FixedWindowChunker(config)
        
        assert chunker.window_size == 120
        assert chunker.overlap_seconds == 10
        assert chunker.min_chunk_length == 20
        assert chunker.config == config
    
    def test_fixed_window_chunker_config_validation_positive_window_size(self):
        """Test configuration validation for positive window size."""
        with pytest.raises(ValueError, match="Window size must be positive"):
            FixedWindowChunker({"window_size": 0})
        
        with pytest.raises(ValueError, match="Window size must be positive"):
            FixedWindowChunker({"window_size": -10})
    
    def test_fixed_window_chunker_config_validation_overlap_seconds(self):
        """Test configuration validation for overlap seconds."""
        with pytest.raises(ValueError, match="Overlap seconds cannot be negative"):
            FixedWindowChunker({"overlap_seconds": -5})
        
        with pytest.raises(ValueError, match="Overlap seconds must be less than window size"):
            FixedWindowChunker({"window_size": 30, "overlap_seconds": 30})
        
        with pytest.raises(ValueError, match="Overlap seconds must be less than window size"):
            FixedWindowChunker({"window_size": 30, "overlap_seconds": 35})
    
    def test_fixed_window_chunker_config_validation_min_chunk_length(self):
        """Test configuration validation for minimum chunk length."""
        with pytest.raises(ValueError, match="Minimum chunk length must be positive"):
            FixedWindowChunker({"min_chunk_length": 0})
        
        with pytest.raises(ValueError, match="Minimum chunk length must be positive"):
            FixedWindowChunker({"min_chunk_length": -5})
    
    def test_fixed_window_chunker_empty_transcript_data(self):
        """Test chunking with empty transcript data."""
        chunker = FixedWindowChunker()
        
        # Test with None
        chunks = chunker.chunk(None)
        assert chunks == []
        
        # Test with empty dict
        chunks = chunker.chunk({})
        assert chunks == []
        
        # Test with no segments
        chunks = chunker.chunk({"segments": []})
        assert chunks == []
    
    def test_fixed_window_chunker_basic_chunking(self):
        """Test basic fixed-window chunking functionality."""
        config = {"window_size": 60, "overlap_seconds": 5}
        chunker = FixedWindowChunker(config)
        
        transcript_data = {
            "source_file": "test_meeting.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Hello everyone, welcome to the meeting."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Thank you Alice. Let's start with the agenda."},
                {"start": 60.0, "end": 90.0, "speaker": "Charlie", "text": "I have some updates to share."},
                {"start": 90.0, "end": 120.0, "speaker": "Alice", "text": "That sounds great, please go ahead."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create 3 chunks: 0-60, 55-115, 110-120
        assert len(chunks) >= 2
        
        # Check first chunk
        first_chunk = chunks[0]
        assert first_chunk.metadata.start_time == 0.0
        assert first_chunk.metadata.end_time == 60.0
        assert first_chunk.metadata.chunk_type == "fixed_window"
        assert first_chunk.metadata.source_file == "test_meeting.json"
        assert "Alice" in first_chunk.content
        assert "Bob" in first_chunk.content
    
    def test_fixed_window_chunker_window_sizes(self):
        """Test chunking with different window sizes."""
        transcript_data = {
            "source_file": "test_meeting.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First segment content."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Second segment content."},
                {"start": 60.0, "end": 90.0, "speaker": "Charlie", "text": "Third segment content."},
                {"start": 90.0, "end": 120.0, "speaker": "Alice", "text": "Fourth segment content."}
            ]
        }
        
        # Test with 30-second windows
        chunker_30 = FixedWindowChunker({"window_size": 30, "overlap_seconds": 0})
        chunks_30 = chunker_30.chunk(transcript_data)
        
        # Should create 4 chunks for 120 seconds with 30-second windows
        assert len(chunks_30) == 4
        
        # Test with 60-second windows
        chunker_60 = FixedWindowChunker({"window_size": 60, "overlap_seconds": 0})
        chunks_60 = chunker_60.chunk(transcript_data)
        
        # Should create 2 chunks for 120 seconds with 60-second windows
        assert len(chunks_60) == 2
    
    def test_fixed_window_chunker_overlap_functionality(self):
        """Test chunking with different overlap configurations."""
        transcript_data = {
            "source_file": "test_meeting.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First segment."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Second segment."},
                {"start": 60.0, "end": 90.0, "speaker": "Charlie", "text": "Third segment."}
            ]
        }
        
        # Test with no overlap
        chunker_no_overlap = FixedWindowChunker({"window_size": 45, "overlap_seconds": 0})
        chunks_no_overlap = chunker_no_overlap.chunk(transcript_data)
        
        # Test with 10-second overlap
        chunker_overlap = FixedWindowChunker({"window_size": 45, "overlap_seconds": 10})
        chunks_overlap = chunker_overlap.chunk(transcript_data)
        
        # Overlap should create more chunks
        assert len(chunks_overlap) >= len(chunks_no_overlap)
        
        # Check that overlapping chunks have overlapping content
        if len(chunks_overlap) >= 2:
            # Verify time overlap
            chunk1_end = chunks_overlap[0].metadata.end_time
            chunk2_start = chunks_overlap[1].metadata.start_time
            assert chunk1_end > chunk2_start  # Should overlap
    
    def test_fixed_window_chunker_boundary_conditions(self):
        """Test chunking boundary conditions."""
        # Test with single segment
        single_segment_data = {
            "source_file": "single_segment.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Single segment content."}
            ]
        }
        
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 5})
        chunks = chunker.chunk(single_segment_data)
        
        assert len(chunks) == 1
        assert chunks[0].metadata.start_time == 0.0
        assert chunks[0].metadata.end_time == 30.0  # Should use actual segment end time
        
        # Test with segment shorter than window
        short_segment_data = {
            "source_file": "short_segment.json",
            "segments": [
                {"start": 0.0, "end": 10.0, "speaker": "Alice", "text": "Short content."}
            ]
        }
        
        chunks = chunker.chunk(short_segment_data)
        assert len(chunks) == 1
        assert chunks[0].metadata.end_time == 10.0
    
    def test_fixed_window_chunker_edge_cases(self):
        """Test edge cases for fixed-window chunking."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 5, "min_chunk_length": 20})
        
        # Test with very short text (should be filtered out)
        short_text_data = {
            "source_file": "short_text.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Hi."}
            ]
        }
        
        chunks = chunker.chunk(short_text_data)
        # Should be filtered out due to min_chunk_length
        assert len(chunks) == 0
        
        # Test with empty text segments
        empty_text_data = {
            "source_file": "empty_text.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": ""},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "   "}
            ]
        }
        
        chunks = chunker.chunk(empty_text_data)
        assert len(chunks) == 0
    
    def test_fixed_window_chunker_speaker_extraction(self):
        """Test speaker extraction from chunks."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 0})
        
        transcript_data = {
            "source_file": "multi_speaker.json",
            "segments": [
                {"start": 0.0, "end": 20.0, "speaker": "Alice", "text": "Alice speaking first."},
                {"start": 20.0, "end": 40.0, "speaker": "Bob", "text": "Bob responding."},
                {"start": 40.0, "end": 60.0, "speaker": "Alice", "text": "Alice speaking again."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Should have both speakers
        assert chunk.metadata.speaker_count == 2
        assert set(chunk.metadata.speakers) == {"Alice", "Bob"}
        
        # Content should include both speakers
        assert "Alice:" in chunk.content
        assert "Bob:" in chunk.content
    
    def test_fixed_window_chunker_chunk_metadata_completeness(self):
        """Test that chunk metadata is complete and accurate."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 5})
        
        transcript_data = {
            "source_file": "metadata_test.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Testing metadata generation."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "This is a test segment."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # With 5-second overlap, should create 2 chunks: [0-60] and [55-60]
        assert len(chunks) == 2
        
        # Check first chunk metadata
        chunk = chunks[0]
        metadata = chunk.metadata
        
        # Check all metadata fields
        assert metadata.chunk_id.startswith("chunk_")
        assert metadata.source_file == "metadata_test.json"
        assert metadata.start_time == 0.0
        assert metadata.end_time == 60.0
        assert metadata.chunk_type == "fixed_window"
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 2
        assert metadata.word_count > 0
        assert isinstance(metadata.created_at, datetime)
    
    def test_fixed_window_chunker_overlapping_segments(self):
        """Test handling of overlapping segments."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 0})
        
        # Create segments that overlap in time
        transcript_data = {
            "source_file": "overlapping.json",
            "segments": [
                {"start": 0.0, "end": 40.0, "speaker": "Alice", "text": "First overlapping segment."},
                {"start": 20.0, "end": 60.0, "speaker": "Bob", "text": "Second overlapping segment."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Both segments should be included
        assert "Alice:" in chunk.content
        assert "Bob:" in chunk.content
        assert chunk.metadata.speaker_count == 2
    
    def test_fixed_window_chunker_segment_overlap_logic(self):
        """Test the segment overlap detection logic."""
        chunker = FixedWindowChunker()
        
        # Test cases for segment overlap
        test_cases = [
            # (seg_start, seg_end, win_start, win_end, expected_overlap)
            (0.0, 30.0, 0.0, 60.0, True),      # Segment within window
            (30.0, 90.0, 0.0, 60.0, True),     # Segment overlaps window
            (0.0, 30.0, 20.0, 80.0, True),     # Segment overlaps window start
            (70.0, 100.0, 20.0, 80.0, True),   # Segment overlaps window end
            (100.0, 130.0, 0.0, 60.0, False),  # Segment after window
            (0.0, 10.0, 50.0, 100.0, False),   # Segment before window
            (20.0, 20.0, 0.0, 60.0, False),    # Zero-length segment
        ]
        
        for seg_start, seg_end, win_start, win_end, expected in test_cases:
            result = chunker._segments_overlap(seg_start, seg_end, win_start, win_end)
            assert result == expected, f"Failed for ({seg_start}, {seg_end}) vs ({win_start}, {win_end})"
    
    def test_fixed_window_chunker_chunk_statistics(self):
        """Test chunk statistics method."""
        config = {
            "window_size": 120,
            "overlap_seconds": 10,
            "min_chunk_length": 25
        }
        chunker = FixedWindowChunker(config)
        
        stats = chunker.get_chunk_statistics()
        
        assert stats["strategy"] == "fixed_window"
        assert stats["window_size"] == 120
        assert stats["overlap_seconds"] == 10
        assert stats["min_chunk_length"] == 25
        assert stats["effective_step_size"] == 110  # 120 - 10
    
    def test_fixed_window_chunker_large_transcript(self):
        """Test chunking with a large transcript."""
        # Create a large transcript with many segments
        segments = []
        for i in range(100):
            segments.append({
                "start": i * 30.0,
                "end": (i + 1) * 30.0,
                "speaker": f"Speaker{i % 3}",
                "text": f"This is segment {i} with some content to test chunking."
            })
        
        transcript_data = {
            "source_file": "large_transcript.json",
            "segments": segments
        }
        
        chunker = FixedWindowChunker({"window_size": 300, "overlap_seconds": 30})
        chunks = chunker.chunk(transcript_data)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # All chunks should have valid metadata
        for chunk in chunks:
            assert chunk.metadata.chunk_id.startswith("chunk_")
            assert chunk.metadata.start_time >= 0
            assert chunk.metadata.end_time > chunk.metadata.start_time
            assert len(chunk.content) > 0
    
    def test_fixed_window_chunker_chunk_uniqueness(self):
        """Test that all generated chunks have unique IDs."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 10})
        
        # Create data that will generate multiple chunks
        segments = []
        for i in range(10):
            segments.append({
                "start": i * 30.0,
                "end": (i + 1) * 30.0,
                "speaker": f"Speaker{i % 2}",
                "text": f"Content for segment {i} with enough text to meet minimum length."
            })
        
        transcript_data = {
            "source_file": "uniqueness_test.json",
            "segments": segments
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Verify all chunk IDs are unique
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
    
    def test_fixed_window_chunker_content_formatting(self):
        """Test chunk content formatting."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 0})
        
        transcript_data = {
            "source_file": "formatting_test.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First statement."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Second statement."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Check content formatting
        expected_content = "Alice: First statement. Bob: Second statement."
        assert chunk.content == expected_content
    
    def test_fixed_window_chunker_missing_speaker_handling(self):
        """Test handling of segments with missing speaker information."""
        chunker = FixedWindowChunker({"window_size": 60, "overlap_seconds": 0})
        
        transcript_data = {
            "source_file": "missing_speaker.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "text": "Statement without speaker."},
                {"start": 30.0, "end": 60.0, "speaker": "Alice", "text": "Statement with speaker."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Should handle missing speaker gracefully
        assert "Unknown:" in chunk.content
        assert "Alice:" in chunk.content
        assert set(chunk.metadata.speakers) == {"Unknown", "Alice"}


class TestFixedWindowChunkerIntegration:
    """Integration tests for FixedWindowChunker."""
    
    def test_fixed_window_chunker_with_config_integration(self):
        """Test integration with configuration loading."""
        # Test with config that matches ChunkingConfig structure
        config = {
            "window_size": 90,
            "overlap_seconds": 15,
            "min_chunk_length": 30
        }
        
        chunker = FixedWindowChunker(config)
        
        # Create realistic transcript data
        transcript_data = {
            "source_file": "integration_test.json",
            "segments": [
                {"start": 0.0, "end": 45.0, "speaker": "Interviewer", "text": "Thank you for joining us today. Can you tell us about your experience?"},
                {"start": 45.0, "end": 120.0, "speaker": "Candidate", "text": "Absolutely. I have been working in software development for over five years, specializing in backend systems and database optimization."},
                {"start": 120.0, "end": 180.0, "speaker": "Interviewer", "text": "That's impressive. Can you walk us through a challenging project you worked on recently?"},
                {"start": 180.0, "end": 300.0, "speaker": "Candidate", "text": "Sure. Last year, I led the migration of our legacy system to a microservices architecture. It involved careful planning and coordination with multiple teams."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create multiple overlapping chunks
        assert len(chunks) >= 2
        
        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_type == "fixed_window"
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == len(chunks)
            assert len(chunk.content) >= 30  # min_chunk_length
    
    def test_fixed_window_chunker_deterministic_behavior(self):
        """Test that chunker produces deterministic results."""
        config = {"window_size": 60, "overlap_seconds": 10}
        
        transcript_data = {
            "source_file": "deterministic_test.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First segment content."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Second segment content."},
                {"start": 60.0, "end": 90.0, "speaker": "Alice", "text": "Third segment content."}
            ]
        }
        
        # Run chunking multiple times
        chunker1 = FixedWindowChunker(config)
        chunks1 = chunker1.chunk(transcript_data)
        
        chunker2 = FixedWindowChunker(config)
        chunks2 = chunker2.chunk(transcript_data)
        
        # Should produce identical results
        assert len(chunks1) == len(chunks2)
        
        for chunk1, chunk2 in zip(chunks1, chunks2):
            assert chunk1.metadata.chunk_id == chunk2.metadata.chunk_id
            assert chunk1.content == chunk2.content
            assert chunk1.metadata.start_time == chunk2.metadata.start_time
            assert chunk1.metadata.end_time == chunk2.metadata.end_time


class TestChunkingConfigurationIntegration:
    """Test suite for chunking configuration integration."""
    
    def test_create_chunker_from_config_fixed_window(self):
        """Test creating FixedWindowChunker from configuration."""
        config = {
            "strategy": "fixed_window",
            "window_size": 90,
            "overlap_seconds": 10,
            "min_chunk_length": 20
        }
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 90
        assert chunker.overlap_seconds == 10
        assert chunker.min_chunk_length == 20
    
    def test_create_chunker_from_config_speaker_based(self):
        """Test creating SpeakerBasedChunker from configuration."""
        config = {
            "strategy": "speaker_based",
            "max_chunk_duration": 180,
            "min_chunk_length": 25,
            "speaker_change_threshold": 2.0,
            "merge_consecutive_same_speaker": False
        }
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, SpeakerBasedChunker)
        assert chunker.max_chunk_duration == 180
        assert chunker.min_chunk_length == 25
        assert chunker.speaker_change_threshold == 2.0
        assert chunker.merge_consecutive_same_speaker == False
    
    def test_create_chunker_from_config_invalid_strategy(self):
        """Test creating chunker with invalid strategy."""
        config = {
            "strategy": "invalid_strategy",
            "window_size": 60
        }
        
        with pytest.raises(ValueError, match="Unsupported chunking strategy"):
            create_chunker_from_config(config)
    
    def test_create_chunker_from_config_default_strategy(self):
        """Test creating chunker with default strategy."""
        config = {}
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 60  # Default value
        assert chunker.overlap_seconds == 5  # Default value
        assert chunker.min_chunk_length == 10  # Default value
    
    def test_get_default_chunking_config(self):
        """Test getting default chunking configuration."""
        config = get_default_chunking_config()
        
        assert config["strategy"] == "fixed_window"
        assert config["window_size"] == 60
        assert config["overlap_seconds"] == 5
        assert config["min_chunk_length"] == 10
        assert config["max_chunk_duration"] == 300
        assert config["speaker_change_threshold"] == 0.0
        assert config["merge_consecutive_same_speaker"] == True
    
    def test_configuration_parameter_validation(self):
        """Test configuration parameter validation."""
        # Test invalid window size
        config = {
            "strategy": "fixed_window",
            "window_size": -10
        }
        
        with pytest.raises(ValueError, match="Window size must be positive"):
            create_chunker_from_config(config)
        
        # Test invalid overlap
        config = {
            "strategy": "fixed_window",
            "window_size": 60,
            "overlap_seconds": -5
        }
        
        with pytest.raises(ValueError, match="Overlap seconds cannot be negative"):
            create_chunker_from_config(config)
        
        # Test invalid max duration
        config = {
            "strategy": "speaker_based",
            "max_chunk_duration": 0
        }
        
        with pytest.raises(ValueError, match="Max chunk duration must be positive"):
            create_chunker_from_config(config)
        
        # Test invalid threshold
        config = {
            "strategy": "speaker_based",
            "speaker_change_threshold": -1.0
        }
        
        with pytest.raises(ValueError, match="Speaker change threshold cannot be negative"):
            create_chunker_from_config(config)
    
    def test_configuration_with_mixed_parameters(self):
        """Test configuration with parameters from both strategies."""
        # This simulates a configuration file that has all parameters
        config = {
            "strategy": "fixed_window",
            "window_size": 90,
            "overlap_seconds": 10,
            "min_chunk_length": 20,
            "max_chunk_duration": 300,
            "speaker_change_threshold": 2.0,
            "merge_consecutive_same_speaker": False
        }
        
        chunker = create_chunker_from_config(config)
        
        # Should create FixedWindowChunker and use relevant parameters
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 90
        assert chunker.overlap_seconds == 10
        assert chunker.min_chunk_length == 20
        
        # Should ignore speaker-based parameters
        assert not hasattr(chunker, 'max_chunk_duration')
        assert not hasattr(chunker, 'speaker_change_threshold')
    
    def test_configuration_file_integration(self):
        """Test integration with configuration file structure."""
        # Simulate configuration from config.py structure
        app_config = {
            "chunking": {
                "strategy": "speaker_based",
                "window_size": 60,
                "overlap_seconds": 5,
                "min_chunk_length": 15,
                "max_chunk_duration": 240,
                "speaker_change_threshold": 1.5,
                "merge_consecutive_same_speaker": True
            }
        }
        
        chunker = create_chunker_from_config(app_config["chunking"])
        
        assert isinstance(chunker, SpeakerBasedChunker)
        assert chunker.max_chunk_duration == 240
        assert chunker.min_chunk_length == 15
        assert chunker.speaker_change_threshold == 1.5
        assert chunker.merge_consecutive_same_speaker == True
    
    def test_configuration_defaults_fallback(self):
        """Test that configuration falls back to defaults when parameters are missing."""
        # Test with partial configuration
        config = {
            "strategy": "fixed_window",
            "window_size": 45
            # Missing overlap_seconds and min_chunk_length
        }
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 45  # Configured value
        assert chunker.overlap_seconds == 5  # Default value
        assert chunker.min_chunk_length == 10  # Default value
    
    def test_configuration_validation_integration(self):
        """Test that configuration validation works with create_chunker_from_config."""
        # This should pass validation
        valid_config = {
            "strategy": "fixed_window",
            "window_size": 60,
            "overlap_seconds": 10,
            "min_chunk_length": 5
        }
        
        chunker = create_chunker_from_config(valid_config)
        assert isinstance(chunker, FixedWindowChunker)
        
        # This should fail validation
        invalid_config = {
            "strategy": "fixed_window",
            "window_size": 30,
            "overlap_seconds": 35  # Greater than window_size
        }
        
        with pytest.raises(ValueError, match="Overlap seconds must be less than window size"):
            create_chunker_from_config(invalid_config)


class TestSpeakerBasedChunker:
    """Test suite for SpeakerBasedChunker class."""
    
    def test_speaker_based_chunker_initialization_default_config(self):
        """Test SpeakerBasedChunker initialization with default configuration."""
        chunker = SpeakerBasedChunker()
        
        assert chunker.max_chunk_duration == 300
        assert chunker.min_chunk_length == 10
        assert chunker.speaker_change_threshold == 0.0
        assert chunker.merge_consecutive_same_speaker == True
        assert chunker.config == {}
    
    def test_speaker_based_chunker_initialization_custom_config(self):
        """Test SpeakerBasedChunker initialization with custom configuration."""
        config = {
            "max_chunk_duration": 180,
            "min_chunk_length": 20,
            "speaker_change_threshold": 2.0,
            "merge_consecutive_same_speaker": False
        }
        
        chunker = SpeakerBasedChunker(config)
        
        assert chunker.max_chunk_duration == 180
        assert chunker.min_chunk_length == 20
        assert chunker.speaker_change_threshold == 2.0
        assert chunker.merge_consecutive_same_speaker == False
        assert chunker.config == config
    
    def test_speaker_based_chunker_config_validation_max_duration(self):
        """Test configuration validation for max chunk duration."""
        with pytest.raises(ValueError, match="Max chunk duration must be positive"):
            SpeakerBasedChunker({"max_chunk_duration": 0})
        
        with pytest.raises(ValueError, match="Max chunk duration must be positive"):
            SpeakerBasedChunker({"max_chunk_duration": -10})
    
    def test_speaker_based_chunker_config_validation_min_length(self):
        """Test configuration validation for minimum chunk length."""
        with pytest.raises(ValueError, match="Minimum chunk length must be positive"):
            SpeakerBasedChunker({"min_chunk_length": 0})
        
        with pytest.raises(ValueError, match="Minimum chunk length must be positive"):
            SpeakerBasedChunker({"min_chunk_length": -5})
    
    def test_speaker_based_chunker_config_validation_threshold(self):
        """Test configuration validation for speaker change threshold."""
        with pytest.raises(ValueError, match="Speaker change threshold cannot be negative"):
            SpeakerBasedChunker({"speaker_change_threshold": -1.0})
    
    def test_speaker_based_chunker_empty_transcript_data(self):
        """Test chunking with empty transcript data."""
        chunker = SpeakerBasedChunker()
        
        # Test with None
        chunks = chunker.chunk(None)
        assert chunks == []
        
        # Test with empty dict
        chunks = chunker.chunk({})
        assert chunks == []
        
        # Test with no segments
        chunks = chunker.chunk({"segments": []})
        assert chunks == []
    
    def test_speaker_based_chunker_single_speaker(self):
        """Test chunking with single speaker."""
        chunker = SpeakerBasedChunker()
        
        transcript_data = {
            "source_file": "single_speaker.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Hello everyone, welcome to the meeting."},
                {"start": 30.0, "end": 60.0, "speaker": "Alice", "text": "Today we'll discuss the quarterly results."},
                {"start": 60.0, "end": 90.0, "speaker": "Alice", "text": "Let's start with the first item on the agenda."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create one chunk for single speaker
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.metadata.start_time == 0.0
        assert chunk.metadata.end_time == 90.0
        assert chunk.metadata.chunk_type == "speaker_based"
        assert chunk.metadata.source_file == "single_speaker.json"
        assert chunk.metadata.speaker_count == 1
        assert chunk.metadata.speakers == ["Alice"]
        assert "Alice:" in chunk.content
    
    def test_speaker_based_chunker_multiple_speakers(self):
        """Test chunking with multiple speakers."""
        chunker = SpeakerBasedChunker()
        
        transcript_data = {
            "source_file": "multi_speaker.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Hello everyone."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Hi Alice, good morning."},
                {"start": 60.0, "end": 90.0, "speaker": "Charlie", "text": "Good morning everyone."},
                {"start": 90.0, "end": 120.0, "speaker": "Alice", "text": "Let's start the meeting."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create 4 chunks for speaker changes
        assert len(chunks) == 4
        
        # Check first chunk (Alice)
        assert chunks[0].metadata.speakers == ["Alice"]
        assert chunks[0].metadata.start_time == 0.0
        assert chunks[0].metadata.end_time == 30.0
        
        # Check second chunk (Bob)
        assert chunks[1].metadata.speakers == ["Bob"]
        assert chunks[1].metadata.start_time == 30.0
        assert chunks[1].metadata.end_time == 60.0
        
        # Check third chunk (Charlie)
        assert chunks[2].metadata.speakers == ["Charlie"]
        assert chunks[2].metadata.start_time == 60.0
        assert chunks[2].metadata.end_time == 90.0
        
        # Check fourth chunk (Alice again)
        assert chunks[3].metadata.speakers == ["Alice"]
        assert chunks[3].metadata.start_time == 90.0
        assert chunks[3].metadata.end_time == 120.0
    
    def test_speaker_based_chunker_merge_consecutive_same_speaker(self):
        """Test merging consecutive segments from same speaker."""
        chunker = SpeakerBasedChunker({"merge_consecutive_same_speaker": True})
        
        transcript_data = {
            "source_file": "consecutive_same.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First part."},
                {"start": 30.0, "end": 60.0, "speaker": "Alice", "text": "Second part."},
                {"start": 60.0, "end": 90.0, "speaker": "Bob", "text": "Bob's response."},
                {"start": 90.0, "end": 120.0, "speaker": "Alice", "text": "Alice again."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should merge consecutive Alice segments
        assert len(chunks) == 3
        
        # First chunk should have both Alice segments
        assert chunks[0].metadata.speakers == ["Alice"]
        assert chunks[0].metadata.start_time == 0.0
        assert chunks[0].metadata.end_time == 60.0
        assert "First part" in chunks[0].content
        assert "Second part" in chunks[0].content
        
        # Second chunk should be Bob
        assert chunks[1].metadata.speakers == ["Bob"]
        assert chunks[1].metadata.start_time == 60.0
        assert chunks[1].metadata.end_time == 90.0
        
        # Third chunk should be Alice again
        assert chunks[2].metadata.speakers == ["Alice"]
        assert chunks[2].metadata.start_time == 90.0
        assert chunks[2].metadata.end_time == 120.0
    
    def test_speaker_based_chunker_no_merge_consecutive_same_speaker(self):
        """Test not merging consecutive segments from same speaker."""
        chunker = SpeakerBasedChunker({"merge_consecutive_same_speaker": False})
        
        transcript_data = {
            "source_file": "no_merge.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First part."},
                {"start": 30.0, "end": 60.0, "speaker": "Alice", "text": "Second part."},
                {"start": 60.0, "end": 90.0, "speaker": "Bob", "text": "Bob's response."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should not merge consecutive Alice segments
        assert len(chunks) == 3
        
        # Each segment should be its own chunk
        assert chunks[0].metadata.start_time == 0.0
        assert chunks[0].metadata.end_time == 30.0
        assert chunks[1].metadata.start_time == 30.0
        assert chunks[1].metadata.end_time == 60.0
        assert chunks[2].metadata.start_time == 60.0
        assert chunks[2].metadata.end_time == 90.0
    
    def test_speaker_based_chunker_max_duration_threshold(self):
        """Test max duration threshold splits long chunks."""
        chunker = SpeakerBasedChunker({"max_chunk_duration": 60})
        
        transcript_data = {
            "source_file": "long_duration.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First part."},
                {"start": 30.0, "end": 60.0, "speaker": "Alice", "text": "Second part."},
                {"start": 60.0, "end": 90.0, "speaker": "Alice", "text": "Third part - should be new chunk."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should split into 2 chunks due to max duration
        assert len(chunks) == 2
        
        # First chunk should be 0-60 seconds
        assert chunks[0].metadata.start_time == 0.0
        assert chunks[0].metadata.end_time == 60.0
        assert chunks[0].metadata.speakers == ["Alice"]
        
        # Second chunk should be 60-90 seconds
        assert chunks[1].metadata.start_time == 60.0
        assert chunks[1].metadata.end_time == 90.0
        assert chunks[1].metadata.speakers == ["Alice"]
    
    def test_speaker_based_chunker_speaker_change_threshold(self):
        """Test speaker change threshold for gaps between segments."""
        chunker = SpeakerBasedChunker({
            "speaker_change_threshold": 5.0,
            "merge_consecutive_same_speaker": True
        })
        
        transcript_data = {
            "source_file": "speaker_gaps.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First statement."},
                {"start": 35.0, "end": 65.0, "speaker": "Alice", "text": "Second statement after gap."},  # 5s gap
                {"start": 67.0, "end": 97.0, "speaker": "Alice", "text": "Third statement short gap."}  # 2s gap
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create 2 chunks: gap >= threshold splits, gap < threshold merges
        assert len(chunks) == 2
        
        # First chunk should be just the first segment
        assert chunks[0].metadata.start_time == 0.0
        assert chunks[0].metadata.end_time == 30.0
        
        # Second chunk should merge the last two segments (gap < threshold)
        assert chunks[1].metadata.start_time == 35.0
        assert chunks[1].metadata.end_time == 97.0
    
    def test_speaker_based_chunker_min_chunk_length_filter(self):
        """Test minimum chunk length filtering."""
        chunker = SpeakerBasedChunker({"min_chunk_length": 20})
        
        transcript_data = {
            "source_file": "short_chunks.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Hi."},  # Too short
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Hello everyone, this is long enough."},
                {"start": 60.0, "end": 90.0, "speaker": "Charlie", "text": "Yes."}  # Too short
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should filter out short chunks
        assert len(chunks) == 1
        assert chunks[0].metadata.speakers == ["Bob"]
        assert len(chunks[0].content) >= 20
    
    def test_speaker_based_chunker_chunk_metadata_completeness(self):
        """Test that chunk metadata is complete and accurate."""
        chunker = SpeakerBasedChunker()
        
        transcript_data = {
            "source_file": "metadata_test.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "Testing metadata generation."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "This is a test segment."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        assert len(chunks) == 2
        
        # Check first chunk metadata
        chunk = chunks[0]
        metadata = chunk.metadata
        
        assert metadata.chunk_id.startswith("chunk_")
        assert metadata.source_file == "metadata_test.json"
        assert metadata.start_time == 0.0
        assert metadata.end_time == 30.0
        assert metadata.chunk_type == "speaker_based"
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 2
        assert metadata.word_count > 0
        assert isinstance(metadata.created_at, datetime)
        assert metadata.speakers == ["Alice"]
        assert metadata.speaker_count == 1
    
    def test_speaker_based_chunker_missing_speaker_handling(self):
        """Test handling of segments with missing speaker information."""
        chunker = SpeakerBasedChunker()
        
        transcript_data = {
            "source_file": "missing_speaker.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "text": "Statement without speaker."},
                {"start": 30.0, "end": 60.0, "speaker": "Alice", "text": "Statement with speaker."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create 2 chunks due to speaker change
        assert len(chunks) == 2
        
        # First chunk should have "Unknown" speaker
        assert chunks[0].metadata.speakers == ["Unknown"]
        assert "Unknown:" in chunks[0].content
        
        # Second chunk should have "Alice" speaker
        assert chunks[1].metadata.speakers == ["Alice"]
        assert "Alice:" in chunks[1].content
    
    def test_speaker_based_chunker_content_formatting(self):
        """Test chunk content formatting."""
        chunker = SpeakerBasedChunker()
        
        transcript_data = {
            "source_file": "formatting_test.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First statement."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Second statement."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        assert len(chunks) == 2
        
        # Check content formatting
        assert chunks[0].content == "Alice: First statement."
        assert chunks[1].content == "Bob: Second statement."
    
    def test_speaker_based_chunker_chunk_uniqueness(self):
        """Test that all generated chunks have unique IDs."""
        chunker = SpeakerBasedChunker()
        
        # Create data with multiple speakers
        segments = []
        for i in range(10):
            segments.append({
                "start": i * 30.0,
                "end": (i + 1) * 30.0,
                "speaker": f"Speaker{i % 3}",
                "text": f"Content for segment {i} with enough text to meet minimum length requirements."
            })
        
        transcript_data = {
            "source_file": "uniqueness_test.json",
            "segments": segments
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Verify all chunk IDs are unique
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
    
    def test_speaker_based_chunker_chunk_statistics(self):
        """Test chunk statistics method."""
        config = {
            "max_chunk_duration": 180,
            "min_chunk_length": 25,
            "speaker_change_threshold": 2.0,
            "merge_consecutive_same_speaker": False
        }
        chunker = SpeakerBasedChunker(config)
        
        stats = chunker.get_chunk_statistics()
        
        assert stats["strategy"] == "speaker_based"
        assert stats["max_chunk_duration"] == 180
        assert stats["min_chunk_length"] == 25
        assert stats["speaker_change_threshold"] == 2.0
        assert stats["merge_consecutive_same_speaker"] == False
    
    def test_speaker_based_chunker_speaker_transitions(self):
        """Test speaker transition detection."""
        chunker = SpeakerBasedChunker()
        
        transcript_data = {
            "source_file": "transitions.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First statement."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Second statement."},
                {"start": 60.0, "end": 90.0, "speaker": "Alice", "text": "Third statement."},
                {"start": 90.0, "end": 120.0, "speaker": "Charlie", "text": "Fourth statement."}
            ]
        }
        
        transitions = chunker.get_speaker_transitions(transcript_data)
        
        # Should detect 3 transitions
        assert len(transitions) == 3
        
        # Check first transition (Alice -> Bob)
        assert transitions[0]["segment_index"] == 1
        assert transitions[0]["time"] == 30.0
        assert transitions[0]["from_speaker"] == "Alice"
        assert transitions[0]["to_speaker"] == "Bob"
        
        # Check second transition (Bob -> Alice)
        assert transitions[1]["segment_index"] == 2
        assert transitions[1]["time"] == 60.0
        assert transitions[1]["from_speaker"] == "Bob"
        assert transitions[1]["to_speaker"] == "Alice"
        
        # Check third transition (Alice -> Charlie)
        assert transitions[2]["segment_index"] == 3
        assert transitions[2]["time"] == 90.0
        assert transitions[2]["from_speaker"] == "Alice"
        assert transitions[2]["to_speaker"] == "Charlie"
    
    def test_speaker_based_chunker_large_transcript(self):
        """Test chunking with a large transcript."""
        chunker = SpeakerBasedChunker({"max_chunk_duration": 120})
        
        # Create a large transcript with many segments
        segments = []
        for i in range(100):
            segments.append({
                "start": i * 30.0,
                "end": (i + 1) * 30.0,
                "speaker": f"Speaker{i % 5}",
                "text": f"This is segment {i} with some content to test speaker-based chunking."
            })
        
        transcript_data = {
            "source_file": "large_transcript.json",
            "segments": segments
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create many chunks
        assert len(chunks) > 10
        
        # All chunks should have valid metadata
        for chunk in chunks:
            assert chunk.metadata.chunk_id.startswith("chunk_")
            assert chunk.metadata.start_time >= 0
            assert chunk.metadata.end_time > chunk.metadata.start_time
            assert len(chunk.content) > 0
            assert len(chunk.metadata.speakers) > 0
    
    def test_speaker_based_chunker_edge_cases(self):
        """Test edge cases for speaker-based chunking."""
        chunker = SpeakerBasedChunker({"min_chunk_length": 20})
        
        # Test with empty text segments
        empty_text_data = {
            "source_file": "empty_text.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": ""},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "   "}
            ]
        }
        
        chunks = chunker.chunk(empty_text_data)
        assert len(chunks) == 0
        
        # Test with very short segments
        short_segments_data = {
            "source_file": "short_segments.json",
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "A"},
                {"start": 1.0, "end": 2.0, "speaker": "Bob", "text": "B"}
            ]
        }
        
        chunks = chunker.chunk(short_segments_data)
        assert len(chunks) == 0  # Both too short


class TestSpeakerBasedChunkerIntegration:
    """Integration tests for SpeakerBasedChunker."""
    
    def test_speaker_based_chunker_with_config_integration(self):
        """Test integration with configuration loading."""
        config = {
            "max_chunk_duration": 120,
            "min_chunk_length": 30,
            "speaker_change_threshold": 1.0,
            "merge_consecutive_same_speaker": True
        }
        
        chunker = SpeakerBasedChunker(config)
        
        # Create realistic transcript data
        transcript_data = {
            "source_file": "integration_test.json",
            "segments": [
                {"start": 0.0, "end": 45.0, "speaker": "Interviewer", "text": "Thank you for joining us today. Can you tell us about your experience?"},
                {"start": 45.0, "end": 120.0, "speaker": "Candidate", "text": "Absolutely. I have been working in software development for over five years."},
                {"start": 125.0, "end": 180.0, "speaker": "Interviewer", "text": "That's impressive. Can you walk us through a challenging project?"},
                {"start": 180.0, "end": 300.0, "speaker": "Candidate", "text": "Sure. Last year, I led the migration of our legacy system to microservices."}
            ]
        }
        
        chunks = chunker.chunk(transcript_data)
        
        # Should create chunks based on speaker changes
        assert len(chunks) >= 3
        
        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_type == "speaker_based"
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == len(chunks)
            assert len(chunk.content) >= 30  # min_chunk_length
    
    def test_speaker_based_chunker_deterministic_behavior(self):
        """Test that chunker produces deterministic results."""
        config = {
            "max_chunk_duration": 120,
            "speaker_change_threshold": 2.0
        }
        
        transcript_data = {
            "source_file": "deterministic_test.json",
            "segments": [
                {"start": 0.0, "end": 30.0, "speaker": "Alice", "text": "First segment content."},
                {"start": 30.0, "end": 60.0, "speaker": "Bob", "text": "Second segment content."},
                {"start": 60.0, "end": 90.0, "speaker": "Alice", "text": "Third segment content."}
            ]
        }
        
        # Run chunking multiple times
        chunker1 = SpeakerBasedChunker(config)
        chunks1 = chunker1.chunk(transcript_data)
        
        chunker2 = SpeakerBasedChunker(config)
        chunks2 = chunker2.chunk(transcript_data)
        
        # Should produce identical results
        assert len(chunks1) == len(chunks2)
        
        for chunk1, chunk2 in zip(chunks1, chunks2):
            assert chunk1.metadata.chunk_id == chunk2.metadata.chunk_id
            assert chunk1.content == chunk2.content
            assert chunk1.metadata.start_time == chunk2.metadata.start_time
            assert chunk1.metadata.end_time == chunk2.metadata.end_time


class TestChunkingConfigurationIntegration:
    """Test suite for chunking configuration integration."""
    
    def test_create_chunker_from_config_fixed_window(self):
        """Test creating FixedWindowChunker from configuration."""
        config = {
            "strategy": "fixed_window",
            "window_size": 90,
            "overlap_seconds": 10,
            "min_chunk_length": 20
        }
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 90
        assert chunker.overlap_seconds == 10
        assert chunker.min_chunk_length == 20
    
    def test_create_chunker_from_config_speaker_based(self):
        """Test creating SpeakerBasedChunker from configuration."""
        config = {
            "strategy": "speaker_based",
            "max_chunk_duration": 180,
            "min_chunk_length": 25,
            "speaker_change_threshold": 2.0,
            "merge_consecutive_same_speaker": False
        }
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, SpeakerBasedChunker)
        assert chunker.max_chunk_duration == 180
        assert chunker.min_chunk_length == 25
        assert chunker.speaker_change_threshold == 2.0
        assert chunker.merge_consecutive_same_speaker == False
    
    def test_create_chunker_from_config_invalid_strategy(self):
        """Test creating chunker with invalid strategy."""
        config = {
            "strategy": "invalid_strategy",
            "window_size": 60
        }
        
        with pytest.raises(ValueError, match="Unsupported chunking strategy"):
            create_chunker_from_config(config)
    
    def test_create_chunker_from_config_default_strategy(self):
        """Test creating chunker with default strategy."""
        config = {}
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 60  # Default value
        assert chunker.overlap_seconds == 5  # Default value
        assert chunker.min_chunk_length == 10  # Default value
    
    def test_get_default_chunking_config(self):
        """Test getting default chunking configuration."""
        config = get_default_chunking_config()
        
        assert config["strategy"] == "fixed_window"
        assert config["window_size"] == 60
        assert config["overlap_seconds"] == 5
        assert config["min_chunk_length"] == 10
        assert config["max_chunk_duration"] == 300
        assert config["speaker_change_threshold"] == 0.0
        assert config["merge_consecutive_same_speaker"] == True
    
    def test_configuration_parameter_validation(self):
        """Test configuration parameter validation."""
        # Test invalid window size
        config = {
            "strategy": "fixed_window",
            "window_size": -10
        }
        
        with pytest.raises(ValueError, match="Window size must be positive"):
            create_chunker_from_config(config)
        
        # Test invalid overlap
        config = {
            "strategy": "fixed_window",
            "window_size": 60,
            "overlap_seconds": -5
        }
        
        with pytest.raises(ValueError, match="Overlap seconds cannot be negative"):
            create_chunker_from_config(config)
        
        # Test invalid max duration
        config = {
            "strategy": "speaker_based",
            "max_chunk_duration": 0
        }
        
        with pytest.raises(ValueError, match="Max chunk duration must be positive"):
            create_chunker_from_config(config)
        
        # Test invalid threshold
        config = {
            "strategy": "speaker_based",
            "speaker_change_threshold": -1.0
        }
        
        with pytest.raises(ValueError, match="Speaker change threshold cannot be negative"):
            create_chunker_from_config(config)
    
    def test_configuration_with_mixed_parameters(self):
        """Test configuration with parameters from both strategies."""
        # This simulates a configuration file that has all parameters
        config = {
            "strategy": "fixed_window",
            "window_size": 90,
            "overlap_seconds": 10,
            "min_chunk_length": 20,
            "max_chunk_duration": 300,
            "speaker_change_threshold": 2.0,
            "merge_consecutive_same_speaker": False
        }
        
        chunker = create_chunker_from_config(config)
        
        # Should create FixedWindowChunker and use relevant parameters
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 90
        assert chunker.overlap_seconds == 10
        assert chunker.min_chunk_length == 20
        
        # Should ignore speaker-based parameters
        assert not hasattr(chunker, 'max_chunk_duration')
        assert not hasattr(chunker, 'speaker_change_threshold')
    
    def test_configuration_file_integration(self):
        """Test integration with configuration file structure."""
        # Simulate configuration from config.py structure
        app_config = {
            "chunking": {
                "strategy": "speaker_based",
                "window_size": 60,
                "overlap_seconds": 5,
                "min_chunk_length": 15,
                "max_chunk_duration": 240,
                "speaker_change_threshold": 1.5,
                "merge_consecutive_same_speaker": True
            }
        }
        
        chunker = create_chunker_from_config(app_config["chunking"])
        
        assert isinstance(chunker, SpeakerBasedChunker)
        assert chunker.max_chunk_duration == 240
        assert chunker.min_chunk_length == 15
        assert chunker.speaker_change_threshold == 1.5
        assert chunker.merge_consecutive_same_speaker == True
    
    def test_configuration_defaults_fallback(self):
        """Test that configuration falls back to defaults when parameters are missing."""
        # Test with partial configuration
        config = {
            "strategy": "fixed_window",
            "window_size": 45
            # Missing overlap_seconds and min_chunk_length
        }
        
        chunker = create_chunker_from_config(config)
        
        assert isinstance(chunker, FixedWindowChunker)
        assert chunker.window_size == 45  # Configured value
        assert chunker.overlap_seconds == 5  # Default value
        assert chunker.min_chunk_length == 10  # Default value
    
    def test_configuration_validation_integration(self):
        """Test that configuration validation works with create_chunker_from_config."""
        # This should pass validation
        valid_config = {
            "strategy": "fixed_window",
            "window_size": 60,
            "overlap_seconds": 10,
            "min_chunk_length": 5
        }
        
        chunker = create_chunker_from_config(valid_config)
        assert isinstance(chunker, FixedWindowChunker)
        
        # This should fail validation
        invalid_config = {
            "strategy": "fixed_window",
            "window_size": 30,
            "overlap_seconds": 35  # Greater than window_size
        }
        
        with pytest.raises(ValueError, match="Overlap seconds must be less than window size"):
            create_chunker_from_config(invalid_config)