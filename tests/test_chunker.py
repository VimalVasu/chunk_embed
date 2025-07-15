"""
Tests for the chunker module, focusing on chunk ID generation and idempotency.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime
from typing import List, Dict, Any

from src.chunker import (
    Chunk, ChunkMetadata, ChunkIDGenerator, BaseChunker, FixedWindowChunker
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