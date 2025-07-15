"""
Transcript loader module for JSON loading and validation.

This module provides functionality to load and validate transcript data from JSON files.
It includes Pydantic models for data validation, JSON Schema enforcement, and an
abstraction layer for future format support (VTT, SRT).
"""

import json
import hashlib
import os
import stat
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator
import jsonschema


class ParticipantModel(BaseModel):
    """Pydantic model for participant data validation."""
    
    id: str = Field(..., description="Unique participant identifier")
    name: str = Field(..., description="Participant display name")
    role: str = Field(..., description="Participant role in the meeting")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        """Validate participant ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("Participant ID must be a non-empty string")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate participant name."""
        if not v or not isinstance(v, str):
            raise ValueError("Participant name must be a non-empty string")
        return v


class TranscriptEntryModel(BaseModel):
    """Pydantic model for individual transcript entry validation."""
    
    speaker_id: str = Field(..., description="ID of the speaker")
    start_time: float = Field(..., description="Start time in seconds from meeting start")
    end_time: float = Field(..., description="End time in seconds from meeting start")
    text: str = Field(..., description="Transcript text content")
    
    @field_validator('start_time')
    @classmethod
    def validate_start_time(cls, v):
        """Validate start_time is non-negative."""
        if v < 0:
            raise ValueError("Start time must be non-negative")
        return v
    
    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v):
        """Validate end_time is non-negative."""
        if v < 0:
            raise ValueError("End time must be non-negative")
        return v
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Validate text content."""
        if not v or not isinstance(v, str):
            raise ValueError("Text must be a non-empty string")
        return v.strip()
    
    @field_validator('end_time')
    @classmethod
    def validate_end_time_after_start(cls, v, info):
        """Validate that end_time is after start_time."""
        if 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError("End time must be after start time")
        return v


class MetadataModel(BaseModel):
    """Pydantic model for transcript metadata validation."""
    
    title: str = Field(..., description="Meeting or transcript title")
    duration_seconds: float = Field(..., description="Total duration in seconds")
    participant_count: int = Field(..., description="Number of participants")
    meeting_id: str = Field(..., description="Unique meeting identifier")
    date: str = Field(..., description="Meeting date in ISO format")
    language: str = Field(default="en", description="Language code")
    
    @field_validator('duration_seconds')
    @classmethod
    def validate_duration(cls, v):
        """Validate duration is positive."""
        if v <= 0:
            raise ValueError("Duration must be positive")
        return v
    
    @field_validator('participant_count')
    @classmethod
    def validate_participant_count(cls, v):
        """Validate participant count is positive."""
        if v <= 0:
            raise ValueError("Participant count must be positive")
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        """Validate date format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Date must be in ISO format")
        return v


class TranscriptDataModel(BaseModel):
    """Pydantic model for complete transcript data validation."""
    
    metadata: MetadataModel = Field(..., description="Transcript metadata")
    participants: List[ParticipantModel] = Field(..., description="List of participants")
    transcript: List[TranscriptEntryModel] = Field(..., description="Transcript entries")
    
    @field_validator('participants')
    @classmethod
    def validate_participants(cls, v):
        """Validate participants list."""
        if not v:
            raise ValueError("Participants list cannot be empty")
        
        # Check for duplicate participant IDs
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate participant IDs found")
        
        return v
    
    @field_validator('transcript')
    @classmethod
    def validate_transcript(cls, v):
        """Validate transcript entries."""
        if not v:
            raise ValueError("Transcript cannot be empty")
        
        # Check start_time ordering
        start_times = [entry.start_time for entry in v]
        if start_times != sorted(start_times):
            raise ValueError("Transcript entries must be ordered by start_time")
        
        return v


class TranscriptLoaderError(Exception):
    """Base exception for transcript loader errors."""
    pass


class ValidationError(TranscriptLoaderError):
    """Exception raised when transcript validation fails."""
    pass


class FileNotFoundError(TranscriptLoaderError):
    """Exception raised when transcript file is not found."""
    pass


class FilePermissionError(TranscriptLoaderError):
    """Exception raised when file access is denied due to permissions."""
    pass


class FileCorruptionError(TranscriptLoaderError):
    """Exception raised when file is corrupted or unreadable."""
    pass


class FileSizeError(TranscriptLoaderError):
    """Exception raised when file size exceeds limits."""
    pass


class FileEncodingError(TranscriptLoaderError):
    """Exception raised when file encoding is invalid."""
    pass


class JSONSchemaError(TranscriptLoaderError):
    """Exception raised when JSON schema validation fails."""
    pass


class AbstractTranscriptLoader(ABC):
    """Abstract base class for transcript loaders."""
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> TranscriptDataModel:
        """Load transcript data from file."""
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate transcript data structure."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        pass


class JSONTranscriptLoader(AbstractTranscriptLoader):
    """JSON transcript loader implementation."""
    
    # Default file size limit: 100MB
    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self, schema_path: Optional[Union[str, Path]] = None, max_file_size: Optional[int] = None):
        """Initialize JSON transcript loader.
        
        Args:
            schema_path: Path to JSON schema file for validation
            max_file_size: Maximum file size in bytes (default: 100MB)
        """
        self.schema_path = schema_path
        self.max_file_size = max_file_size or self.DEFAULT_MAX_FILE_SIZE
        self._schema = None
        self._load_schema()
    
    def _load_schema(self):
        """Load JSON schema for validation."""
        if self.schema_path:
            schema_file = Path(self.schema_path)
            if schema_file.exists():
                try:
                    with open(schema_file, 'r') as f:
                        self._schema = json.load(f)
                except json.JSONDecodeError as e:
                    raise JSONSchemaError(f"Invalid JSON schema file: {e}")
            else:
                raise FileNotFoundError(f"Schema file not found: {schema_file}")
        else:
            # Use default schema path
            default_schema = Path(__file__).parent.parent / "test_data" / "schemas" / "transcript_schema.json"
            if default_schema.exists():
                try:
                    with open(default_schema, 'r') as f:
                        self._schema = json.load(f)
                except json.JSONDecodeError as e:
                    raise JSONSchemaError(f"Invalid default schema file: {e}")
    
    def load(self, file_path: Union[str, Path]) -> TranscriptDataModel:
        """Load transcript data from JSON file with comprehensive error handling.
        
        Args:
            file_path: Path to the JSON transcript file
            
        Returns:
            TranscriptDataModel: Validated transcript data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            FilePermissionError: If file access is denied
            FileSizeError: If file size exceeds limits
            FileCorruptionError: If file is corrupted or unreadable
            FileEncodingError: If file encoding is invalid
            ValidationError: If the data doesn't pass validation
            JSONSchemaError: If JSON schema validation fails
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Transcript file not found: {file_path}. "
                f"Please check the file path and ensure the file exists."
            )
        
        # Check if path is actually a file
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Path exists but is not a file: {file_path}. "
                f"Please provide a path to a valid JSON file."
            )
        
        # Check file permissions
        try:
            if not os.access(file_path, os.R_OK):
                raise FilePermissionError(
                    f"Permission denied: Cannot read file {file_path}. "
                    f"Please check file permissions and ensure read access."
                )
        except OSError as e:
            raise FilePermissionError(
                f"Error checking file permissions for {file_path}: {e}"
            )
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise FileSizeError(
                    f"File size ({file_size:,} bytes) exceeds maximum allowed size "
                    f"({self.max_file_size:,} bytes) for {file_path}. "
                    f"Consider splitting the file or increasing the size limit."
                )
            
            if file_size == 0:
                raise FileCorruptionError(
                    f"File is empty: {file_path}. "
                    f"Please provide a valid JSON file with transcript data."
                )
        except OSError as e:
            raise FileCorruptionError(
                f"Error accessing file {file_path}: {e}. "
                f"File may be corrupted or inaccessible."
            )
        
        # Check file extension
        if file_path.suffix.lower() not in ['.json']:
            # Warning but not error - allow flexibility
            pass
        
        # Attempt to read and parse the file
        try:
            # Try multiple encodings in case of encoding issues
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            data = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        
                        # Check for null bytes or other corruption indicators
                        if '\x00' in content:
                            raise FileCorruptionError(
                                f"File contains null bytes and may be corrupted: {file_path}"
                            )
                        
                        # Attempt to parse JSON
                        data = json.loads(content)
                        used_encoding = encoding
                        break
                        
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError as e:
                    # If it's the UTF-8 attempt, provide detailed error
                    if encoding == 'utf-8':
                        raise ValidationError(
                            f"Invalid JSON in file {file_path} (line {e.lineno}, column {e.colno}): {e.msg}. "
                            f"Please check the JSON syntax and ensure the file is properly formatted."
                        )
                    continue
            
            if data is None:
                raise FileEncodingError(
                    f"Could not decode file {file_path} with any of the supported encodings "
                    f"({', '.join(encodings)}). Please ensure the file is properly encoded."
                )
            
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in file {file_path} (line {e.lineno}, column {e.colno}): {e.msg}. "
                f"Please check the JSON syntax and ensure the file is properly formatted."
            )
        except FileCorruptionError:
            # Re-raise file corruption errors as-is
            raise
        except FileEncodingError:
            # Re-raise encoding errors as-is
            raise
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except PermissionError as e:
            raise FilePermissionError(
                f"Permission denied while reading file {file_path}: {e}. "
                f"Please check file permissions and ensure read access."
            )
        except OSError as e:
            raise FileCorruptionError(
                f"System error while reading file {file_path}: {e}. "
                f"File may be corrupted, locked, or on an inaccessible drive."
            )
        except MemoryError:
            raise FileSizeError(
                f"Not enough memory to load file {file_path}. "
                f"File may be too large for available system memory."
            )
        except Exception as e:
            raise TranscriptLoaderError(
                f"Unexpected error while reading file {file_path}: {e}. "
                f"Please check the file and try again."
            )
        
        # Validate data structure at high level
        if not isinstance(data, dict):
            raise ValidationError(
                f"JSON file {file_path} must contain a JSON object at the root level, "
                f"but found {type(data).__name__}. Please ensure the file contains a valid transcript object."
            )
        
        # Perform JSON schema validation if available
        if self._schema:
            try:
                jsonschema.validate(data, self._schema)
            except jsonschema.ValidationError as e:
                raise JSONSchemaError(
                    f"JSON schema validation failed for {file_path}: {e.message}. "
                    f"Please ensure the file structure matches the expected schema."
                )
            except Exception as e:
                raise JSONSchemaError(
                    f"Error during JSON schema validation for {file_path}: {e}"
                )
        
        # Perform integrity checks
        try:
            self._perform_integrity_checks(data, file_path)
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            raise ValidationError(
                f"Integrity check failed for {file_path}: {e}"
            )
        
        # Validate using Pydantic models
        try:
            return TranscriptDataModel(**data)
        except Exception as e:
            # Extract more specific error information from Pydantic ValidationError
            error_details = str(e)
            
            # Provide context-specific error messages for common issues
            if "duration_seconds" in error_details and "positive" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Duration must be positive. "
                    f"Found duration_seconds with invalid value. "
                    f"Please ensure the meeting duration is greater than 0."
                )
            elif "participant_count" in error_details and "positive" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Participant count must be positive. "
                    f"Please ensure participant_count is greater than 0."
                )
            elif "date" in error_details and "ISO format" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Date must be in ISO format. "
                    f"Please use format like '2023-01-01T10:00:00Z' or '2023-01-01T10:00:00+00:00'."
                )
            elif "End time must be after start time" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Found transcript entry where end_time is not after start_time. "
                    f"Please ensure all transcript entries have end_time > start_time."
                )
            elif "Transcript entries must be ordered by start_time" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Transcript entries are not in chronological order. "
                    f"Please sort all transcript entries by start_time in ascending order."
                )
            elif "Duplicate participant IDs found" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Found duplicate participant IDs. "
                    f"Please ensure all participant IDs are unique."
                )
            elif "empty" in error_details.lower():
                if "participants" in error_details:
                    raise ValidationError(
                        f"Data validation failed for {file_path}: Participants list cannot be empty. "
                        f"Please include at least one participant in the participants array."
                    )
                elif "transcript" in error_details:
                    raise ValidationError(
                        f"Data validation failed for {file_path}: Transcript cannot be empty. "
                        f"Please include at least one transcript entry."
                    )
                else:
                    raise ValidationError(
                        f"Data validation failed for {file_path}: Required field cannot be empty. "
                        f"Error: {error_details}"
                    )
            elif "non-empty string" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Found empty or invalid string field. "
                    f"All text fields (name, id, text, etc.) must be non-empty strings. "
                    f"Error: {error_details}"
                )
            elif "non-negative" in error_details:
                raise ValidationError(
                    f"Data validation failed for {file_path}: Found negative timestamp. "
                    f"All start_time and end_time values must be non-negative. "
                    f"Error: {error_details}"
                )
            else:
                # Generic error message with original error details
                raise ValidationError(
                    f"Data validation failed for {file_path}: {error_details}. "
                    f"Please check that all required fields are present and correctly formatted. "
                    f"Common issues: missing required fields, incorrect data types, invalid values."
                )
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate transcript data structure.
        
        Args:
            data: Dictionary containing transcript data
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # JSON schema validation
            if self._schema:
                try:
                    jsonschema.validate(data, self._schema)
                except jsonschema.ValidationError as e:
                    raise JSONSchemaError(f"JSON schema validation failed: {e.message}")
            
            # Pydantic model validation
            TranscriptDataModel(**data)
            return True
        except (JSONSchemaError, ValidationError):
            # Re-raise these as-is
            raise
        except Exception as e:
            raise ValidationError(f"Validation failed: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.json']
    
    def _perform_integrity_checks(self, data: Dict[str, Any], file_path: Path):
        """Perform integrity checks on transcript data.
        
        Args:
            data: Dictionary containing transcript data
            file_path: Path to the source file for error reporting
            
        Raises:
            ValidationError: If integrity checks fail
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Data must be a dictionary in {file_path}")
        
        # Check required top-level keys
        required_keys = {'metadata', 'participants', 'transcript'}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValidationError(
                f"Missing required keys in {file_path}: {sorted(missing_keys)}. "
                f"A valid transcript file must contain 'metadata', 'participants', and 'transcript' sections. "
                f"Please add the missing sections to your JSON file."
            )
        
        # Check participant consistency
        if 'participants' in data and 'transcript' in data:
            participant_ids = {p.get('id') for p in data.get('participants', [])}
            transcript_speaker_ids = {entry.get('speaker_id') for entry in data.get('transcript', [])}
            
            invalid_speakers = transcript_speaker_ids - participant_ids
            if invalid_speakers:
                raise ValidationError(
                    f"Invalid speaker IDs in transcript {file_path}: {invalid_speakers}. "
                    f"All speaker IDs must be defined in participants list."
                )
        
        # Check start_time ordering
        if 'transcript' in data:
            transcript_entries = data['transcript']
            if transcript_entries:
                start_times = [entry.get('start_time', 0) for entry in transcript_entries]
                if start_times != sorted(start_times):
                    # Find the first out-of-order entry for better error context
                    for i in range(1, len(start_times)):
                        if start_times[i] < start_times[i-1]:
                            raise ValidationError(
                                f"Transcript entries must be ordered by start_time in {file_path}. "
                                f"Found entry at index {i} with start_time {start_times[i]}s "
                                f"that comes before previous entry with start_time {start_times[i-1]}s. "
                                f"Please sort all transcript entries by start_time in ascending order."
                            )
                    # Fallback if we can't find specific entries
                    raise ValidationError(
                        f"Transcript entries must be ordered by start_time in {file_path}. "
                        f"Please sort all transcript entries by start_time in ascending order."
                    )
        
        # Check duration consistency
        if 'metadata' in data and 'transcript' in data:
            metadata_duration = data['metadata'].get('duration_seconds', 0)
            transcript_entries = data['transcript']
            
            if transcript_entries:
                last_entry = transcript_entries[-1]
                last_end_time = last_entry.get('end_time', 0)
                actual_duration = last_end_time
                
                # Allow 5% tolerance for duration mismatch
                tolerance = metadata_duration * 0.05
                if abs(actual_duration - metadata_duration) > tolerance:
                    raise ValidationError(
                        f"Duration mismatch in {file_path}: metadata says {metadata_duration}s, "
                        f"but transcript ends at {actual_duration}s. "
                        f"Please ensure the metadata duration_seconds matches the actual transcript duration, "
                        f"or verify that the last transcript entry's end_time is correct."
                    )


class TranscriptLoaderFactory:
    """Factory for creating appropriate transcript loaders."""
    
    _loaders = {
        '.json': JSONTranscriptLoader,
    }
    
    @classmethod
    def create_loader(cls, file_path: Union[str, Path], **kwargs) -> AbstractTranscriptLoader:
        """Create appropriate loader for the given file format.
        
        Args:
            file_path: Path to the transcript file
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            AbstractTranscriptLoader: Appropriate loader instance
            
        Raises:
            TranscriptLoaderError: If format is not supported
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in cls._loaders:
            raise TranscriptLoaderError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {list(cls._loaders.keys())}"
            )
        
        loader_class = cls._loaders[file_extension]
        return loader_class(**kwargs)
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of all supported file formats."""
        return list(cls._loaders.keys())


def load_transcript(file_path: Union[str, Path], **kwargs) -> TranscriptDataModel:
    """Convenience function to load transcript data.
    
    Args:
        file_path: Path to the transcript file
        **kwargs: Additional arguments to pass to the loader
        
    Returns:
        TranscriptDataModel: Validated transcript data
    """
    loader = TranscriptLoaderFactory.create_loader(file_path, **kwargs)
    return loader.load(file_path)


def validate_transcript_data(data: Dict[str, Any]) -> bool:
    """Convenience function to validate transcript data.
    
    Args:
        data: Dictionary containing transcript data
        
    Returns:
        bool: True if data is valid
    """
    loader = JSONTranscriptLoader()
    return loader.validate(data)