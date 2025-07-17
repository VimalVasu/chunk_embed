"""
Chunking module for transcript data processing.
Implements various chunking strategies with hash-based ID generation for idempotency.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


@dataclass
class ChunkMetadata:
    """Metadata associated with a chunk."""
    chunk_id: str
    source_file: str
    start_time: float
    end_time: float
    speaker_count: int
    speakers: List[str]
    word_count: int
    created_at: datetime = field(default_factory=datetime.now)
    chunk_type: str = "unknown"
    chunk_index: int = 0
    total_chunks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker_count": self.speaker_count,
            "speakers": self.speakers,
            "word_count": self.word_count,
            "created_at": self.created_at.isoformat(),
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks
        }


@dataclass
class Chunk:
    """Represents a chunk of transcript data."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict()
        }


class ChunkIDGenerator:
    """
    Generates deterministic, hash-based IDs for chunks to ensure idempotency.
    
    The ID generation scheme uses SHA-256 hashing of normalized content
    along with key metadata to create unique, reproducible identifiers.
    """
    
    @staticmethod
    def generate_chunk_id(
        content: str,
        source_file: str,
        start_time: float,
        end_time: float,
        chunk_type: str = "unknown",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a deterministic chunk ID using hash-based approach.
        
        Args:
            content: The text content of the chunk
            source_file: Source file path/identifier
            start_time: Start timestamp of the chunk
            end_time: End timestamp of the chunk
            chunk_type: Type of chunking strategy used
            additional_context: Additional context for ID generation
            
        Returns:
            str: Unique, deterministic chunk ID
        """
        # Normalize content for consistent hashing
        normalized_content = ChunkIDGenerator._normalize_content(content)
        
        # Create consistent data structure for hashing
        hash_data = {
            "content": normalized_content,
            "source_file": source_file,
            "start_time": round(start_time, 3),  # Round to milliseconds
            "end_time": round(end_time, 3),
            "chunk_type": chunk_type
        }
        
        # Add additional context if provided
        if additional_context:
            hash_data["context"] = additional_context
        
        # Create deterministic JSON string
        json_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        hash_bytes = hashlib.sha256(json_str.encode('utf-8')).digest()
        
        # Convert to hex and take first 16 characters for readability
        chunk_id = hash_bytes.hex()[:16]
        
        return f"chunk_{chunk_id}"
    
    @staticmethod
    def _normalize_content(content: str) -> str:
        """
        Normalize content for consistent hashing.
        
        Args:
            content: Raw content string
            
        Returns:
            str: Normalized content string
        """
        # Remove extra whitespace and normalize line endings
        normalized = ' '.join(content.split())
        
        # Convert to lowercase for case-insensitive comparison
        normalized = normalized.lower()
        
        # Remove common punctuation that might vary
        normalized = normalized.replace('"', '').replace("'", "")
        
        return normalized.strip()
    
    @staticmethod
    def detect_collision(chunk_id: str, existing_chunks: List[Chunk]) -> bool:
        """
        Detect if a chunk ID collides with existing chunks.
        
        Args:
            chunk_id: The chunk ID to check
            existing_chunks: List of existing chunks
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        return any(chunk.metadata.chunk_id == chunk_id for chunk in existing_chunks)
    
    @staticmethod
    def handle_collision(
        content: str,
        source_file: str,
        start_time: float,
        end_time: float,
        chunk_type: str,
        collision_counter: int = 1
    ) -> str:
        """
        Handle chunk ID collision by adding a collision counter.
        
        Args:
            content: The text content of the chunk
            source_file: Source file path/identifier
            start_time: Start timestamp of the chunk
            end_time: End timestamp of the chunk
            chunk_type: Type of chunking strategy used
            collision_counter: Counter for handling collisions
            
        Returns:
            str: New chunk ID with collision handling
        """
        additional_context = {"collision_counter": collision_counter}
        
        return ChunkIDGenerator.generate_chunk_id(
            content=content,
            source_file=source_file,
            start_time=start_time,
            end_time=end_time,
            chunk_type=chunk_type,
            additional_context=additional_context
        )


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary for chunking parameters
        """
        self.config = config or {}
        self._chunk_id_generator = ChunkIDGenerator()
        self._chunk_registry = {}  # Track generated chunks for duplicate prevention
    
    @abstractmethod
    def chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Abstract method to chunk transcript data.
        
        Args:
            transcript_data: Transcript data in standard format
            
        Returns:
            List[Chunk]: List of chunks with metadata
        """
        pass
    
    def _create_chunk_metadata(
        self,
        content: str,
        source_file: str,
        start_time: float,
        end_time: float,
        speakers: List[str],
        chunk_type: str,
        chunk_index: int = 0,
        total_chunks: int = 0
    ) -> ChunkMetadata:
        """
        Create chunk metadata with generated ID.
        
        Args:
            content: Chunk content
            source_file: Source file identifier
            start_time: Start timestamp
            end_time: End timestamp
            speakers: List of speakers in chunk
            chunk_type: Type of chunking strategy
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            
        Returns:
            ChunkMetadata: Complete metadata object
        """
        chunk_id = self._chunk_id_generator.generate_chunk_id(
            content=content,
            source_file=source_file,
            start_time=start_time,
            end_time=end_time,
            chunk_type=chunk_type
        )
        
        return ChunkMetadata(
            chunk_id=chunk_id,
            source_file=source_file,
            start_time=start_time,
            end_time=end_time,
            speaker_count=len(set(speakers)),
            speakers=list(set(speakers)),
            word_count=len(content.split()),
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            total_chunks=total_chunks
        )
    
    def _validate_chunk_uniqueness(self, chunks: List[Chunk]) -> bool:
        """
        Validate that all chunks have unique IDs.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            bool: True if all chunks have unique IDs
        """
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        return len(chunk_ids) == len(set(chunk_ids))
    
    def _detect_duplicate_chunk(self, chunk: Chunk) -> bool:
        """
        Detect if a chunk is a duplicate based on ID and content.
        
        Args:
            chunk: Chunk to check for duplication
            
        Returns:
            bool: True if chunk is a duplicate
        """
        chunk_id = chunk.metadata.chunk_id
        
        # Check if chunk ID already exists in registry
        if chunk_id in self._chunk_registry:
            existing_chunk = self._chunk_registry[chunk_id]
            
            # Compare content to confirm it's truly a duplicate
            if chunk.content == existing_chunk.content:
                return True
            else:
                # Same ID but different content - this is a collision
                # Handle by regenerating ID
                return False
        
        return False
    
    def _prevent_duplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Prevent duplicate chunks by filtering and resolving collisions.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            List[Chunk]: List of unique chunks
        """
        unique_chunks = []
        collision_counters = {}
        
        for chunk in chunks:
            # Check if this chunk is a duplicate
            if self._detect_duplicate_chunk(chunk):
                # Skip duplicate chunks
                continue
            
            # Handle ID collisions
            original_id = chunk.metadata.chunk_id
            collision_counter = collision_counters.get(original_id, 0)
            
            # If collision counter exists, we need to regenerate the ID
            if collision_counter > 0:
                new_id = self._chunk_id_generator.handle_collision(
                    content=chunk.content,
                    source_file=chunk.metadata.source_file,
                    start_time=chunk.metadata.start_time,
                    end_time=chunk.metadata.end_time,
                    chunk_type=chunk.metadata.chunk_type,
                    collision_counter=collision_counter
                )
                chunk.metadata.chunk_id = new_id
            
            # Update collision counter
            collision_counters[original_id] = collision_counter + 1
            
            # Add to registry and unique chunks
            self._chunk_registry[chunk.metadata.chunk_id] = chunk
            unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _is_chunk_content_duplicate(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """
        Check if two chunks have duplicate content based on normalized comparison.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            bool: True if chunks have duplicate content
        """
        # Normalize content for comparison
        normalized_content1 = self._chunk_id_generator._normalize_content(chunk1.content)
        normalized_content2 = self._chunk_id_generator._normalize_content(chunk2.content)
        
        return normalized_content1 == normalized_content2
    
    def _detect_semantic_duplicates(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Detect semantically duplicate chunks with similar content.
        
        Args:
            chunks: List of chunks to check for semantic duplicates
            
        Returns:
            List[Chunk]: List of chunks with semantic duplicates removed
        """
        unique_chunks = []
        
        for chunk in chunks:
            is_duplicate = False
            
            # Check against all previously added chunks
            for existing_chunk in unique_chunks:
                if self._is_chunk_content_duplicate(chunk, existing_chunk):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _validate_chunk_integrity(self, chunks: List[Chunk]) -> bool:
        """
        Validate the integrity of chunks (no duplicates, valid IDs, etc.).
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            bool: True if all chunks pass integrity checks
        """
        if not chunks:
            return True
        
        # Check for unique IDs
        if not self._validate_chunk_uniqueness(chunks):
            return False
        
        # Check for valid chunk IDs
        for chunk in chunks:
            if not chunk.metadata.chunk_id.startswith("chunk_"):
                return False
            
            if len(chunk.metadata.chunk_id) != 22:  # "chunk_" + 16 hex chars
                return False
        
        # Check for valid timestamps
        for chunk in chunks:
            if chunk.metadata.start_time >= chunk.metadata.end_time:
                return False
        
        return True
    
    def _clear_chunk_registry(self):
        """Clear the chunk registry for fresh chunking operations."""
        self._chunk_registry.clear()
    
    def _get_chunk_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the chunk registry.
        
        Returns:
            Dict[str, Any]: Registry statistics
        """
        return {
            "total_chunks": len(self._chunk_registry),
            "chunk_ids": list(self._chunk_registry.keys()),
            "memory_usage_bytes": sum(
                len(chunk.content.encode('utf-8')) for chunk in self._chunk_registry.values()
            )
        }


class FixedWindowChunker(BaseChunker):
    """
    Fixed-window chunking strategy that splits transcripts into fixed-size time windows.
    
    This chunker creates chunks based on configurable time windows with optional overlap.
    It ensures chunks maintain speaker continuity and provides comprehensive metadata.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fixed-window chunker with configuration.
        
        Args:
            config: Configuration dictionary with chunking parameters
        """
        super().__init__(config)
        
        # Set configuration values with defaults
        self.window_size = self.config.get("window_size", 60)  # seconds
        self.overlap_seconds = self.config.get("overlap_seconds", 5)  # seconds
        self.min_chunk_length = self.config.get("min_chunk_length", 10)  # characters
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate chunking configuration parameters."""
        if self.window_size <= 0:
            raise ValueError("Window size must be positive")
        
        if self.overlap_seconds < 0:
            raise ValueError("Overlap seconds cannot be negative")
        
        if self.overlap_seconds >= self.window_size:
            raise ValueError("Overlap seconds must be less than window size")
        
        if self.min_chunk_length <= 0:
            raise ValueError("Minimum chunk length must be positive")
    
    def chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Split transcript into fixed-size time windows.
        
        Args:
            transcript_data: Dictionary containing transcript with segments
            
        Returns:
            List[Chunk]: List of chunks with metadata
        """
        if not transcript_data or "segments" not in transcript_data:
            return []
        
        segments = transcript_data["segments"]
        source_file = transcript_data.get("source_file", "unknown")
        
        if not segments:
            return []
        
        # Calculate total duration and chunk boundaries
        total_duration = self._get_total_duration(segments)
        chunk_boundaries = self._calculate_chunk_boundaries(total_duration)
        
        chunks = []
        for i, (start_time, end_time) in enumerate(chunk_boundaries):
            # Extract segments for this time window
            window_segments = self._extract_segments_for_window(
                segments, start_time, end_time
            )
            
            if not window_segments:
                continue
            
            # Create chunk content and metadata
            chunk_content = self._create_chunk_content(window_segments)
            
            # Skip chunks that are too short
            if len(chunk_content) < self.min_chunk_length:
                continue
            
            speakers = self._extract_speakers(window_segments)
            
            metadata = self._create_chunk_metadata(
                content=chunk_content,
                source_file=source_file,
                start_time=start_time,
                end_time=end_time,
                speakers=speakers,
                chunk_type="fixed_window",
                chunk_index=i,
                total_chunks=len(chunk_boundaries)
            )
            
            chunks.append(Chunk(content=chunk_content, metadata=metadata))
        
        # Apply duplicate detection and prevention
        chunks = self._prevent_duplicate_chunks(chunks)
        
        # Validate chunk integrity
        if not self._validate_chunk_integrity(chunks):
            raise ValueError("Chunk integrity validation failed")
        
        return chunks
    
    def _get_total_duration(self, segments: List[Dict[str, Any]]) -> float:
        """
        Calculate the total duration of the transcript.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            float: Total duration in seconds
        """
        if not segments:
            return 0.0
        
        return max(segment.get("end", 0.0) for segment in segments)
    
    def _calculate_chunk_boundaries(self, total_duration: float) -> List[tuple]:
        """
        Calculate chunk boundaries based on window size and overlap.
        
        Args:
            total_duration: Total duration of the transcript
            
        Returns:
            List[tuple]: List of (start_time, end_time) tuples
        """
        boundaries = []
        step_size = self.window_size - self.overlap_seconds
        
        current_start = 0.0
        while current_start < total_duration:
            current_end = min(current_start + self.window_size, total_duration)
            boundaries.append((current_start, current_end))
            
            # Move to next window
            current_start += step_size
            
            # Prevent infinite loop for edge cases
            if current_start >= total_duration:
                break
        
        return boundaries
    
    def _extract_segments_for_window(
        self, 
        segments: List[Dict[str, Any]], 
        start_time: float, 
        end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Extract segments that overlap with the given time window.
        
        Args:
            segments: List of transcript segments
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            List[Dict[str, Any]]: Segments within the time window
        """
        window_segments = []
        
        for segment in segments:
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)
            
            # Check if segment overlaps with the window
            if self._segments_overlap(segment_start, segment_end, start_time, end_time):
                window_segments.append(segment)
        
        return window_segments
    
    def _segments_overlap(
        self, 
        seg_start: float, 
        seg_end: float, 
        win_start: float, 
        win_end: float
    ) -> bool:
        """
        Check if a segment overlaps with a time window.
        
        Args:
            seg_start: Segment start time
            seg_end: Segment end time
            win_start: Window start time
            win_end: Window end time
            
        Returns:
            bool: True if segment overlaps with window
        """
        # Handle zero-length segments
        if seg_start == seg_end:
            return False
        
        return seg_start < win_end and seg_end > win_start
    
    def _create_chunk_content(self, segments: List[Dict[str, Any]]) -> str:
        """
        Create chunk content from segments.
        
        Args:
            segments: List of segments for the chunk
            
        Returns:
            str: Formatted chunk content
        """
        content_parts = []
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            
            if text:
                content_parts.append(f"{speaker}: {text}")
        
        return " ".join(content_parts)
    
    def _extract_speakers(self, segments: List[Dict[str, Any]]) -> List[str]:
        """
        Extract unique speakers from segments.
        
        Args:
            segments: List of segments
            
        Returns:
            List[str]: List of unique speakers
        """
        speakers = set()
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            speakers.add(speaker)
        
        return sorted(list(speakers))
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the chunking configuration.
        
        Returns:
            Dict[str, Any]: Configuration statistics
        """
        return {
            "strategy": "fixed_window",
            "window_size": self.window_size,
            "overlap_seconds": self.overlap_seconds,
            "min_chunk_length": self.min_chunk_length,
            "effective_step_size": self.window_size - self.overlap_seconds
        }


class SpeakerBasedChunker(BaseChunker):
    """
    Speaker-based chunking strategy that creates chunks based on speaker changes.
    
    This chunker groups consecutive segments by the same speaker, with configurable
    thresholds for maximum chunk duration and minimum chunk length.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize speaker-based chunker with configuration.
        
        Args:
            config: Configuration dictionary with chunking parameters
        """
        super().__init__(config)
        
        # Set configuration values with defaults
        self.max_chunk_duration = self.config.get("max_chunk_duration", 300)  # seconds
        self.min_chunk_length = self.config.get("min_chunk_length", 10)  # characters
        self.speaker_change_threshold = self.config.get("speaker_change_threshold", 0.0)  # seconds
        self.merge_consecutive_same_speaker = self.config.get("merge_consecutive_same_speaker", True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate chunking configuration parameters."""
        if self.max_chunk_duration <= 0:
            raise ValueError("Max chunk duration must be positive")
        
        if self.min_chunk_length <= 0:
            raise ValueError("Minimum chunk length must be positive")
        
        if self.speaker_change_threshold < 0:
            raise ValueError("Speaker change threshold cannot be negative")
    
    def chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Split transcript based on speaker changes.
        
        Args:
            transcript_data: Dictionary containing transcript with segments
            
        Returns:
            List[Chunk]: List of chunks with metadata
        """
        if not transcript_data or "segments" not in transcript_data:
            return []
        
        segments = transcript_data["segments"]
        source_file = transcript_data.get("source_file", "unknown")
        
        if not segments:
            return []
        
        # Group segments by speaker transitions
        speaker_groups = self._group_segments_by_speaker(segments)
        
        # Create chunks from speaker groups
        chunks = []
        for i, group in enumerate(speaker_groups):
            chunk_content = self._create_chunk_content(group["segments"])
            
            # Skip chunks that are too short
            if len(chunk_content) < self.min_chunk_length:
                continue
            
            metadata = self._create_chunk_metadata(
                content=chunk_content,
                source_file=source_file,
                start_time=group["start_time"],
                end_time=group["end_time"],
                speakers=group["speakers"],
                chunk_type="speaker_based",
                chunk_index=i,
                total_chunks=len(speaker_groups)
            )
            
            chunks.append(Chunk(content=chunk_content, metadata=metadata))
        
        # Apply duplicate detection and prevention
        chunks = self._prevent_duplicate_chunks(chunks)
        
        # Update total_chunks count after filtering
        for i, chunk in enumerate(chunks):
            chunk.metadata.chunk_index = i
            chunk.metadata.total_chunks = len(chunks)
        
        # Validate chunk integrity
        if not self._validate_chunk_integrity(chunks):
            raise ValueError("Chunk integrity validation failed")
        
        return chunks
    
    def _group_segments_by_speaker(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group consecutive segments by speaker.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List[Dict[str, Any]]: List of speaker groups with metadata
        """
        if not segments:
            return []
        
        groups = []
        current_group = {
            "segments": [],
            "start_time": None,
            "end_time": None,
            "speakers": set(),
            "duration": 0.0
        }
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)
            
            # Check if we should start a new group
            should_start_new_group = self._should_start_new_group(
                current_group, speaker, segment_start, segment_end
            )
            
            if should_start_new_group and current_group["segments"]:
                # Finalize current group
                groups.append(self._finalize_group(current_group))
                current_group = {
                    "segments": [],
                    "start_time": None,
                    "end_time": None,
                    "speakers": set(),
                    "duration": 0.0
                }
            
            # Add segment to current group
            current_group["segments"].append(segment)
            current_group["speakers"].add(speaker)
            
            if current_group["start_time"] is None:
                current_group["start_time"] = segment_start
            
            current_group["end_time"] = segment_end
            current_group["duration"] = current_group["end_time"] - current_group["start_time"]
        
        # Finalize the last group
        if current_group["segments"]:
            groups.append(self._finalize_group(current_group))
        
        return groups
    
    def _should_start_new_group(
        self, 
        current_group: Dict[str, Any], 
        speaker: str, 
        segment_start: float, 
        segment_end: float
    ) -> bool:
        """
        Determine if a new group should be started.
        
        Args:
            current_group: Current group being built
            speaker: Speaker of the current segment
            segment_start: Start time of current segment
            segment_end: End time of current segment
            
        Returns:
            bool: True if new group should be started
        """
        # First segment always starts a new group
        if not current_group["segments"]:
            return False
        
        # Check if duration would exceed maximum
        potential_duration = segment_end - current_group["start_time"]
        if potential_duration > self.max_chunk_duration:
            return True
        
        # Get the last speaker in the current group
        last_speaker = current_group["segments"][-1].get("speaker", "Unknown")
        
        # Check speaker change
        if speaker != last_speaker:
            # Always start new group on speaker change
            return True
        
        # If merge_consecutive_same_speaker is False, start new group for each segment
        if not self.merge_consecutive_same_speaker:
            return True
        
        # Check for significant gap between segments of the same speaker
        if self.speaker_change_threshold > 0:
            last_segment_end = current_group["segments"][-1].get("end", 0.0)
            gap = segment_start - last_segment_end
            if gap >= self.speaker_change_threshold:
                return True
        
        return False
    
    def _finalize_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize a speaker group.
        
        Args:
            group: Group to finalize
            
        Returns:
            Dict[str, Any]: Finalized group with metadata
        """
        return {
            "segments": group["segments"],
            "start_time": group["start_time"],
            "end_time": group["end_time"],
            "speakers": sorted(list(group["speakers"])),
            "duration": group["duration"]
        }
    
    def _create_chunk_content(self, segments: List[Dict[str, Any]]) -> str:
        """
        Create chunk content from segments.
        
        Args:
            segments: List of segments for the chunk
            
        Returns:
            str: Formatted chunk content
        """
        content_parts = []
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            
            if text:
                content_parts.append(f"{speaker}: {text}")
        
        return " ".join(content_parts)
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the chunking configuration.
        
        Returns:
            Dict[str, Any]: Configuration statistics
        """
        return {
            "strategy": "speaker_based",
            "max_chunk_duration": self.max_chunk_duration,
            "min_chunk_length": self.min_chunk_length,
            "speaker_change_threshold": self.speaker_change_threshold,
            "merge_consecutive_same_speaker": self.merge_consecutive_same_speaker
        }
    
    def get_speaker_transitions(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get information about speaker transitions in the transcript.
        
        Args:
            transcript_data: Dictionary containing transcript with segments
            
        Returns:
            List[Dict[str, Any]]: List of speaker transition information
        """
        if not transcript_data or "segments" not in transcript_data:
            return []
        
        segments = transcript_data["segments"]
        if not segments:
            return []
        
        transitions = []
        previous_speaker = None
        
        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", "Unknown")
            start_time = segment.get("start", 0.0)
            
            if previous_speaker is not None and speaker != previous_speaker:
                transitions.append({
                    "segment_index": i,
                    "time": start_time,
                    "from_speaker": previous_speaker,
                    "to_speaker": speaker
                })
            
            previous_speaker = speaker
        
        return transitions


def create_chunker_from_config(config: Dict[str, Any]) -> BaseChunker:
    """
    Create a chunker instance from configuration.
    
    Args:
        config: Configuration dictionary with chunking parameters
        
    Returns:
        BaseChunker: Configured chunker instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    strategy = config.get("strategy", "fixed_window")
    
    if strategy == "fixed_window":
        return FixedWindowChunker(config)
    elif strategy == "speaker_based":
        return SpeakerBasedChunker(config)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")


def get_default_chunking_config() -> Dict[str, Any]:
    """
    Get default chunking configuration.
    
    Returns:
        Dict[str, Any]: Default chunking configuration
    """
    return {
        "strategy": "fixed_window",
        "window_size": 60,
        "overlap_seconds": 5,
        "min_chunk_length": 10,
        "max_chunk_duration": 300,
        "speaker_change_threshold": 0.0,
        "merge_consecutive_same_speaker": True
    }