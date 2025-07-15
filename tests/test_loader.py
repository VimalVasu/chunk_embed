"""
Test loader module for Task 3.1
Tests basic module structure and functionality
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.loader import (
    TranscriptDataModel,
    ParticipantModel,
    TranscriptEntryModel,
    MetadataModel,
    JSONTranscriptLoader,
    AbstractTranscriptLoader,
    TranscriptLoaderFactory,
    TranscriptLoaderError,
    ValidationError,
    FileNotFoundError,
    FilePermissionError,
    FileCorruptionError,
    FileSizeError,
    FileEncodingError,
    JSONSchemaError,
    load_transcript,
    validate_transcript_data
)


class TestLoaderModule:
    """Test that loader module has correct structure and imports."""
    
    def test_module_imports(self):
        """Test that all required classes and functions can be imported."""
        # Test that main classes exist
        assert TranscriptDataModel is not None
        assert ParticipantModel is not None
        assert TranscriptEntryModel is not None
        assert MetadataModel is not None
        assert JSONTranscriptLoader is not None
        assert AbstractTranscriptLoader is not None
        assert TranscriptLoaderFactory is not None
        
        # Test that exceptions exist
        assert TranscriptLoaderError is not None
        assert ValidationError is not None
        assert FileNotFoundError is not None
        assert FilePermissionError is not None
        assert FileCorruptionError is not None
        assert FileSizeError is not None
        assert FileEncodingError is not None
        assert JSONSchemaError is not None
        
        # Test that utility functions exist
        assert load_transcript is not None
        assert validate_transcript_data is not None
    
    def test_exception_hierarchy(self):
        """Test that exception hierarchy is correct."""
        # Test inheritance
        assert issubclass(ValidationError, TranscriptLoaderError)
        assert issubclass(FileNotFoundError, TranscriptLoaderError)
        assert issubclass(FilePermissionError, TranscriptLoaderError)
        assert issubclass(FileCorruptionError, TranscriptLoaderError)
        assert issubclass(FileSizeError, TranscriptLoaderError)
        assert issubclass(FileEncodingError, TranscriptLoaderError)
        assert issubclass(JSONSchemaError, TranscriptLoaderError)
        assert issubclass(TranscriptLoaderError, Exception)
    
    def test_abstract_loader_interface(self):
        """Test that AbstractTranscriptLoader is properly abstract."""
        from abc import ABC
        
        # Test that it's abstract
        assert issubclass(AbstractTranscriptLoader, ABC)
        
        # Test that it can't be instantiated directly
        with pytest.raises(TypeError):
            AbstractTranscriptLoader()
    
    def test_json_loader_inheritance(self):
        """Test that JSONTranscriptLoader inherits from AbstractTranscriptLoader."""
        assert issubclass(JSONTranscriptLoader, AbstractTranscriptLoader)
    
    def test_pydantic_models_exist(self):
        """Test that Pydantic models are properly defined."""
        from pydantic import BaseModel
        
        # Test inheritance
        assert issubclass(ParticipantModel, BaseModel)
        assert issubclass(TranscriptEntryModel, BaseModel)
        assert issubclass(MetadataModel, BaseModel)
        assert issubclass(TranscriptDataModel, BaseModel)
    
    def test_loader_factory_class_exists(self):
        """Test that TranscriptLoaderFactory class exists and has required methods."""
        # Test class exists
        assert TranscriptLoaderFactory is not None
        
        # Test required methods exist
        assert hasattr(TranscriptLoaderFactory, 'create_loader')
        assert hasattr(TranscriptLoaderFactory, 'get_supported_formats')
        assert callable(TranscriptLoaderFactory.create_loader)
        assert callable(TranscriptLoaderFactory.get_supported_formats)
    
    def test_convenience_functions_exist(self):
        """Test that convenience functions exist and are callable."""
        assert callable(load_transcript)
        assert callable(validate_transcript_data)
    
    def test_module_docstring(self):
        """Test that module has proper docstring."""
        import src.loader as loader_module
        
        assert loader_module.__doc__ is not None
        assert "Transcript loader module" in loader_module.__doc__
        assert "JSON loading and validation" in loader_module.__doc__


class TestJSONTranscriptLoader:
    """Test JSONTranscriptLoader basic functionality."""
    
    def test_loader_initialization(self):
        """Test that JSONTranscriptLoader can be initialized."""
        loader = JSONTranscriptLoader()
        assert loader is not None
        assert isinstance(loader, JSONTranscriptLoader)
        assert isinstance(loader, AbstractTranscriptLoader)
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"type": "object"}')
    def test_loader_initialization_with_schema(self, mock_file, mock_exists):
        """Test that JSONTranscriptLoader can be initialized with schema path."""
        loader = JSONTranscriptLoader(schema_path="test_schema.json")
        assert loader is not None
        assert loader.schema_path == "test_schema.json"
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_loader_initialization_with_missing_schema(self, mock_exists):
        """Test that JSONTranscriptLoader raises error for missing schema."""
        with pytest.raises(FileNotFoundError):
            JSONTranscriptLoader(schema_path="missing_schema.json")
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    def test_loader_initialization_with_invalid_schema(self, mock_file, mock_exists):
        """Test that JSONTranscriptLoader raises error for invalid schema."""
        with pytest.raises(JSONSchemaError):
            JSONTranscriptLoader(schema_path="invalid_schema.json")
    
    def test_get_supported_formats(self):
        """Test that get_supported_formats returns correct formats."""
        loader = JSONTranscriptLoader()
        formats = loader.get_supported_formats()
        
        assert isinstance(formats, list)
        assert '.json' in formats
        assert len(formats) > 0
    
    def test_loader_has_required_methods(self):
        """Test that loader has all required methods."""
        loader = JSONTranscriptLoader()
        
        # Test required methods from abstract class
        assert hasattr(loader, 'load')
        assert hasattr(loader, 'validate')
        assert hasattr(loader, 'get_supported_formats')
        assert callable(loader.load)
        assert callable(loader.validate)
        assert callable(loader.get_supported_formats)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=False)
    def test_file_not_found_error(self, mock_exists, mock_file):
        """Test that FileNotFoundError is raised for non-existent files."""
        loader = JSONTranscriptLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("non_existent_file.json")
    
    def test_validate_method_exists(self):
        """Test that validate method exists and works with basic data."""
        loader = JSONTranscriptLoader()
        
        # Test with empty data (should fail)
        with pytest.raises((ValidationError, JSONSchemaError)):
            loader.validate({})
    
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_invalid_json(self, mock_exists, mock_file):
        """Test that ValidationError is raised for invalid JSON."""
        loader = JSONTranscriptLoader()
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(ValidationError) as exc_info:
                loader.load("invalid.json")
        assert "Invalid JSON" in str(exc_info.value)
    
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_file_io_error(self, mock_exists, mock_file):
        """Test that TranscriptLoaderError is raised for file I/O errors."""
        loader = JSONTranscriptLoader()
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(TranscriptLoaderError) as exc_info:
                loader.load("permission_denied.json")
        assert "Error reading file" in str(exc_info.value)
    
    @patch('pathlib.Path.exists', return_value=False)  # No schema file available
    def test_validate_with_valid_data(self, mock_exists):
        """Test validate method with valid data."""
        loader = JSONTranscriptLoader()  # No schema, so only Pydantic validation
        
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Should not raise exception
        assert loader.validate(valid_data) is True
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"type": "object"}')
    def test_validate_with_schema(self, mock_file, mock_exists):
        """Test validate method with schema validation."""
        loader = JSONTranscriptLoader(schema_path="test_schema.json")
        
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Should not raise exception
        assert loader.validate(valid_data) is True
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_with_valid_data(self, mock_exists, mock_file):
        """Test load method with valid data."""
        valid_json = json.dumps({
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        })
        
        mock_file.return_value.read.return_value = valid_json
        
        # Load without schema validation by setting no schema file
        loader = JSONTranscriptLoader()
        
        # Now mock the file existence for the load method
        with patch('pathlib.Path.exists', return_value=True):
            result = loader.load("valid.json")
        
        assert isinstance(result, TranscriptDataModel)
        assert result.metadata.title == "Test Meeting"
        assert len(result.participants) == 1
        assert len(result.transcript) == 1
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_load_with_integrity_check_failures(self, mock_file, mock_exists):
        """Test load method with integrity check failures."""
        # Test missing required keys
        invalid_json = json.dumps({
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            }
            # Missing participants and transcript
        })
        
        mock_file.return_value.read.return_value = invalid_json
        
        loader = JSONTranscriptLoader()
        with pytest.raises(ValidationError) as exc_info:
            loader.load("invalid.json")
        assert "Missing required keys" in str(exc_info.value)
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_load_with_speaker_consistency_error(self, mock_file, mock_exists):
        """Test load method with speaker consistency error."""
        invalid_json = json.dumps({
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p2", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}  # p2 not in participants
            ]
        })
        
        mock_file.return_value.read.return_value = invalid_json
        
        loader = JSONTranscriptLoader()
        with pytest.raises(ValidationError) as exc_info:
            loader.load("invalid.json")
        assert "Invalid speaker IDs" in str(exc_info.value)
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_load_with_timestamp_ordering_error(self, mock_file, mock_exists):
        """Test load method with timestamp ordering error."""
        invalid_json = json.dumps({
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 13.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 10.0, "end_time": 13.0, "text": "Second"},
                {"speaker_id": "p1", "start_time": 5.0, "end_time": 8.0, "text": "First"}  # Out of order
            ]
        })
        
        mock_file.return_value.read.return_value = invalid_json
        
        loader = JSONTranscriptLoader()
        with pytest.raises(ValidationError) as exc_info:
            loader.load("invalid.json")
        assert "must be ordered by start_time" in str(exc_info.value)
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_load_with_duration_mismatch_error(self, mock_file, mock_exists):
        """Test load method with duration mismatch error."""
        invalid_json = json.dumps({
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 100.0,  # Too short
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"},
                {"speaker_id": "p1", "start_time": 200.0, "end_time": 210.0, "text": "Goodbye"}  # Ends at 210s
            ]
        })
        
        mock_file.return_value.read.return_value = invalid_json
        
        loader = JSONTranscriptLoader()
        with pytest.raises(ValidationError) as exc_info:
            loader.load("invalid.json")
        assert "Duration mismatch" in str(exc_info.value)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_with_non_dict_data(self, mock_exists, mock_file):
        """Test load method with non-dict data."""
        invalid_json = json.dumps("not a dict")
        
        mock_file.return_value.read.return_value = invalid_json
        
        loader = JSONTranscriptLoader()
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(ValidationError) as exc_info:
                loader.load("invalid.json")
        assert "Data must be a dictionary" in str(exc_info.value)


class TestTranscriptLoaderFactory:
    """Test TranscriptLoaderFactory functionality."""
    
    def test_get_supported_formats(self):
        """Test that factory returns supported formats."""
        formats = TranscriptLoaderFactory.get_supported_formats()
        
        assert isinstance(formats, list)
        assert '.json' in formats
        assert len(formats) > 0
    
    def test_create_loader_json(self):
        """Test that factory creates JSONTranscriptLoader for .json files."""
        loader = TranscriptLoaderFactory.create_loader("test.json")
        
        assert isinstance(loader, JSONTranscriptLoader)
        assert isinstance(loader, AbstractTranscriptLoader)
    
    def test_create_loader_unsupported_format(self):
        """Test that factory raises error for unsupported formats."""
        with pytest.raises(TranscriptLoaderError):
            TranscriptLoaderFactory.create_loader("test.txt")
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"type": "object"}')
    def test_create_loader_with_kwargs(self, mock_file, mock_exists):
        """Test that factory passes kwargs to loader constructor."""
        loader = TranscriptLoaderFactory.create_loader(
            "test.json",
            schema_path="custom_schema.json"
        )
        
        assert isinstance(loader, JSONTranscriptLoader)
        assert loader.schema_path == "custom_schema.json"


class TestPydanticModels:
    """Test Pydantic model structure and comprehensive validation."""
    
    def test_participant_model_structure(self):
        """Test ParticipantModel structure."""
        # Test required fields
        participant_data = {
            "id": "participant_1",
            "name": "John Doe",
            "role": "participant"
        }
        
        participant = ParticipantModel(**participant_data)
        assert participant.id == "participant_1"
        assert participant.name == "John Doe"
        assert participant.role == "participant"
    
    def test_participant_model_with_role(self):
        """Test ParticipantModel with different role values."""
        participant_data = {
            "id": "participant_1",
            "name": "John Doe",
            "role": "moderator"
        }
        
        participant = ParticipantModel(**participant_data)
        assert participant.role == "moderator"
    
    def test_participant_model_validation_errors(self):
        """Test ParticipantModel validation errors."""
        from pydantic import ValidationError as PydanticValidationError
        
        # Test missing required fields
        with pytest.raises(PydanticValidationError):
            ParticipantModel()
        
        with pytest.raises(PydanticValidationError):
            ParticipantModel(id="test")  # Missing name and role
        
        with pytest.raises(PydanticValidationError):
            ParticipantModel(name="test")  # Missing id and role
        
        # Test empty string id validation
        with pytest.raises(PydanticValidationError) as exc_info:
            ParticipantModel(id="", name="John Doe", role="participant")
        assert "Participant ID must be a non-empty string" in str(exc_info.value)
        
        # Test empty string name validation
        with pytest.raises(PydanticValidationError) as exc_info:
            ParticipantModel(id="test", name="", role="participant")
        assert "Participant name must be a non-empty string" in str(exc_info.value)
        
        # Test non-string id validation
        with pytest.raises(PydanticValidationError) as exc_info:
            ParticipantModel(id=123, name="John Doe", role="participant")
        # Pydantic first checks type, then our custom validator
        assert "Input should be a valid string" in str(exc_info.value)
        
        # Test non-string name validation
        with pytest.raises(PydanticValidationError) as exc_info:
            ParticipantModel(id="test", name=123, role="participant")
        # Pydantic first checks type, then our custom validator
        assert "Input should be a valid string" in str(exc_info.value)
    
    def test_transcript_entry_model_structure(self):
        """Test TranscriptEntryModel structure."""
        entry_data = {
            "speaker_id": "participant_1",
            "start_time": 10.5,
            "end_time": 12.8,
            "text": "Hello everyone"
        }
        
        entry = TranscriptEntryModel(**entry_data)
        assert entry.speaker_id == "participant_1"
        assert entry.start_time == 10.5
        assert entry.end_time == 12.8
        assert entry.text == "Hello everyone"
    
    def test_transcript_entry_model_validation_errors(self):
        """Test TranscriptEntryModel validation errors."""
        from pydantic import ValidationError as PydanticValidationError
        
        # Test missing required fields
        with pytest.raises(PydanticValidationError):
            TranscriptEntryModel()
        
        # Test negative start_time
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptEntryModel(
                speaker_id="test",
                start_time=-1.0,
                end_time=2.0,
                text="Hello"
            )
        assert "Start time must be non-negative" in str(exc_info.value)
        
        # Test negative end_time
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptEntryModel(
                speaker_id="test",
                start_time=0.0,
                end_time=-1.0,
                text="Hello"
            )
        assert "End time must be non-negative" in str(exc_info.value)
        
        # Test end_time before start_time
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptEntryModel(
                speaker_id="test",
                start_time=5.0,
                end_time=2.0,
                text="Hello"
            )
        assert "End time must be after start time" in str(exc_info.value)
        
        # Test empty text
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptEntryModel(
                speaker_id="test",
                start_time=0.0,
                end_time=1.0,
                text=""
            )
        assert "Text must be a non-empty string" in str(exc_info.value)
        
        # Test non-string text
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptEntryModel(
                speaker_id="test",
                start_time=0.0,
                end_time=1.0,
                text=123
            )
        # Pydantic first checks type, then our custom validator
        assert "Input should be a valid string" in str(exc_info.value)
    
    def test_transcript_entry_text_strip(self):
        """Test that text is stripped of whitespace."""
        entry = TranscriptEntryModel(
            speaker_id="test",
            start_time=0.0,
            end_time=1.0,
            text="  Hello world  "
        )
        assert entry.text == "Hello world"
    
    def test_metadata_model_structure(self):
        """Test MetadataModel structure."""
        metadata_data = {
            "title": "Test Meeting",
            "duration_seconds": 300.0,
            "participant_count": 2,
            "meeting_id": "meeting_123",
            "date": "2023-01-01T10:00:00Z"
        }
        
        metadata = MetadataModel(**metadata_data)
        assert metadata.title == "Test Meeting"
        assert metadata.duration_seconds == 300.0
        assert metadata.participant_count == 2
        assert metadata.meeting_id == "meeting_123"
        assert metadata.date == "2023-01-01T10:00:00Z"
        assert metadata.language == "en"  # default value
    
    def test_metadata_model_validation_errors(self):
        """Test MetadataModel validation errors."""
        from pydantic import ValidationError as PydanticValidationError
        
        # Test missing required fields
        with pytest.raises(PydanticValidationError):
            MetadataModel()
        
        # Test negative duration
        with pytest.raises(PydanticValidationError) as exc_info:
            MetadataModel(
                title="Test",
                duration_seconds=-1.0,
                participant_count=1,
                meeting_id="test",
                date="2023-01-01T10:00:00Z"
            )
        assert "Duration must be positive" in str(exc_info.value)
        
        # Test zero duration
        with pytest.raises(PydanticValidationError) as exc_info:
            MetadataModel(
                title="Test",
                duration_seconds=0.0,
                participant_count=1,
                meeting_id="test",
                date="2023-01-01T10:00:00Z"
            )
        assert "Duration must be positive" in str(exc_info.value)
        
        # Test negative participant count
        with pytest.raises(PydanticValidationError) as exc_info:
            MetadataModel(
                title="Test",
                duration_seconds=300.0,
                participant_count=-1,
                meeting_id="test",
                date="2023-01-01T10:00:00Z"
            )
        assert "Participant count must be positive" in str(exc_info.value)
        
        # Test zero participant count
        with pytest.raises(PydanticValidationError) as exc_info:
            MetadataModel(
                title="Test",
                duration_seconds=300.0,
                participant_count=0,
                meeting_id="test",
                date="2023-01-01T10:00:00Z"
            )
        assert "Participant count must be positive" in str(exc_info.value)
        
        # Test invalid date format
        with pytest.raises(PydanticValidationError) as exc_info:
            MetadataModel(
                title="Test",
                duration_seconds=300.0,
                participant_count=1,
                meeting_id="test",
                date="invalid-date"
            )
        assert "Date must be in ISO format" in str(exc_info.value)
    
    def test_metadata_model_date_formats(self):
        """Test various valid date formats."""
        valid_dates = [
            "2023-01-01T10:00:00Z",
            "2023-01-01T10:00:00+00:00",
            "2023-01-01T10:00:00",
            "2023-01-01T10:00:00.123Z",
            "2023-12-31T23:59:59Z"
        ]
        
        for date_str in valid_dates:
            metadata = MetadataModel(
                title="Test",
                duration_seconds=300.0,
                participant_count=1,
                meeting_id="test",
                date=date_str
            )
            assert metadata.date == date_str
    
    def test_transcript_data_model_structure(self):
        """Test TranscriptDataModel structure."""
        # Create minimal valid data
        data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 300.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z"
            },
            "participants": [
                {
                    "id": "participant_1",
                    "name": "John Doe",
                    "role": "participant"
                }
            ],
            "transcript": [
                {
                    "speaker_id": "participant_1",
                    "start_time": 10.5,
                    "end_time": 12.8,
                    "text": "Hello everyone"
                }
            ]
        }
        
        transcript = TranscriptDataModel(**data)
        assert transcript.metadata.title == "Test Meeting"
        assert len(transcript.participants) == 1
        assert len(transcript.transcript) == 1
        assert transcript.participants[0].name == "John Doe"
        assert transcript.transcript[0].text == "Hello everyone"
    
    def test_transcript_data_model_validation_errors(self):
        """Test TranscriptDataModel validation errors."""
        from pydantic import ValidationError as PydanticValidationError
        
        # Test missing required fields
        with pytest.raises(PydanticValidationError):
            TranscriptDataModel()
        
        base_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 300.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z"
            },
            "participants": [
                {
                    "id": "participant_1",
                    "name": "John Doe",
                    "role": "participant"
                }
            ],
            "transcript": [
                {
                    "speaker_id": "participant_1",
                    "start_time": 10.5,
                    "end_time": 12.8,
                    "text": "Hello everyone"
                }
            ]
        }
        
        # Test empty participants list
        invalid_data = base_data.copy()
        invalid_data["participants"] = []
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptDataModel(**invalid_data)
        assert "Participants list cannot be empty" in str(exc_info.value)
        
        # Test empty transcript list
        invalid_data = base_data.copy()
        invalid_data["transcript"] = []
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptDataModel(**invalid_data)
        assert "Transcript cannot be empty" in str(exc_info.value)
        
        # Test duplicate participant IDs
        invalid_data = base_data.copy()
        invalid_data["participants"] = [
            {"id": "participant_1", "name": "John Doe", "role": "participant"},
            {"id": "participant_1", "name": "Jane Doe", "role": "participant"}
        ]
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptDataModel(**invalid_data)
        assert "Duplicate participant IDs found" in str(exc_info.value)
        
        # Test unordered timestamps
        invalid_data = base_data.copy()
        invalid_data["transcript"] = [
            {"speaker_id": "participant_1", "start_time": 20.0, "end_time": 22.0, "text": "Second"},
            {"speaker_id": "participant_1", "start_time": 10.0, "end_time": 12.0, "text": "First"}
        ]
        with pytest.raises(PydanticValidationError) as exc_info:
            TranscriptDataModel(**invalid_data)
        assert "Transcript entries must be ordered by start_time" in str(exc_info.value)
    
    def test_transcript_data_model_complex_scenarios(self):
        """Test TranscriptDataModel with complex valid scenarios."""
        # Test with multiple participants and entries
        data = {
            "metadata": {
                "title": "Complex Meeting",
                "duration_seconds": 600.0,
                "participant_count": 3,
                "meeting_id": "complex_meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "fr"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "host"},
                {"id": "p2", "name": "Bob", "role": "participant"},
                {"id": "p3", "name": "Charlie", "role": "guest"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Welcome everyone"},
                {"speaker_id": "p2", "start_time": 5.0, "end_time": 7.0, "text": "Thank you"},
                {"speaker_id": "p3", "start_time": 10.0, "end_time": 14.0, "text": "Great to be here"},
                {"speaker_id": "p1", "start_time": 15.0, "end_time": 16.0, "text": "Perfect"}
            ]
        }
        
        transcript = TranscriptDataModel(**data)
        assert len(transcript.participants) == 3
        assert len(transcript.transcript) == 4
        assert transcript.metadata.language == "fr"
        assert transcript.participants[0].role == "host"
        assert transcript.participants[1].role == "participant"
        assert transcript.participants[2].role == "guest"
    
    def test_pydantic_model_serialization(self):
        """Test that Pydantic models can be serialized and deserialized."""
        # Create a complete model
        data = {
            "metadata": {
                "title": "Serialization Test",
                "duration_seconds": 300.0,
                "participant_count": 1,
                "meeting_id": "serial_test",
                "date": "2023-01-01T10:00:00Z"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "host"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Create model instance
        original = TranscriptDataModel(**data)
        
        # Serialize to dict
        serialized = original.model_dump()
        
        # Deserialize back
        deserialized = TranscriptDataModel(**serialized)
        
        # Verify they're equivalent
        assert original.metadata.title == deserialized.metadata.title
        assert original.participants[0].name == deserialized.participants[0].name
        assert original.transcript[0].text == deserialized.transcript[0].text
    
    def test_pydantic_model_json_serialization(self):
        """Test that Pydantic models can be JSON serialized."""
        data = {
            "metadata": {
                "title": "JSON Test",
                "duration_seconds": 300.0,
                "participant_count": 1,
                "meeting_id": "json_test",
                "date": "2023-01-01T10:00:00Z"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Create model instance
        original = TranscriptDataModel(**data)
        
        # Serialize to JSON
        json_str = original.model_dump_json()
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["metadata"]["title"] == "JSON Test"
        assert parsed["participants"][0]["name"] == "Alice"
        assert parsed["transcript"][0]["text"] == "Hello"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_transcript_function_exists(self):
        """Test that load_transcript function exists."""
        assert load_transcript is not None
        assert callable(load_transcript)
    
    def test_validate_transcript_data_function_exists(self):
        """Test that validate_transcript_data function exists."""
        assert validate_transcript_data is not None
        assert callable(validate_transcript_data)
    
    def test_validate_transcript_data_with_invalid_data(self):
        """Test validate_transcript_data with invalid data."""
        # Test with empty data (should raise ValidationError or JSONSchemaError)
        with pytest.raises((ValidationError, JSONSchemaError)):
            validate_transcript_data({})
    
    @patch('pathlib.Path.exists', return_value=False)  # No schema file available
    def test_validate_transcript_data_with_valid_data(self, mock_exists):
        """Test validate_transcript_data with valid data."""
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Should not raise exception
        assert validate_transcript_data(valid_data) is True
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_transcript_with_valid_data(self, mock_exists, mock_file):
        """Test load_transcript convenience function with valid data."""
        valid_json = json.dumps({
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        })
        
        mock_file.return_value.read.return_value = valid_json
        
        with patch('pathlib.Path.exists', return_value=True):
            result = load_transcript("valid.json")
        
        assert isinstance(result, TranscriptDataModel)
        assert result.metadata.title == "Test Meeting"
        assert len(result.participants) == 1
        assert len(result.transcript) == 1
    
    def test_load_transcript_with_unsupported_format(self):
        """Test load_transcript with unsupported format."""
        with pytest.raises(TranscriptLoaderError):
            load_transcript("test.txt")
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"type": "object"}')
    def test_load_transcript_with_kwargs(self, mock_file, mock_exists):
        """Test load_transcript with kwargs passed to loader."""
        valid_json = json.dumps({
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        })
        
        mock_file.return_value.read.return_value = valid_json
        
        result = load_transcript("valid.json", schema_path="test_schema.json")
        
        assert isinstance(result, TranscriptDataModel)
        assert result.metadata.title == "Test Meeting"


class TestErrorHandling:
    """Test error handling and custom exceptions."""
    
    def test_custom_exceptions_can_be_raised(self):
        """Test that custom exceptions can be raised and caught."""
        # Test TranscriptLoaderError
        with pytest.raises(TranscriptLoaderError):
            raise TranscriptLoaderError("Test error")
        
        # Test ValidationError
        with pytest.raises(ValidationError):
            raise ValidationError("Test validation error")
        
        # Test FileNotFoundError
        with pytest.raises(FileNotFoundError):
            raise FileNotFoundError("Test file not found")
        
        # Test JSONSchemaError
        with pytest.raises(JSONSchemaError):
            raise JSONSchemaError("Test schema error")
    
    def test_exception_messages(self):
        """Test that exception messages are preserved."""
        test_message = "Test error message"
        
        try:
            raise TranscriptLoaderError(test_message)
        except TranscriptLoaderError as e:
            assert str(e) == test_message
        
        try:
            raise ValidationError(test_message)
        except ValidationError as e:
            assert str(e) == test_message


class TestJSONSchemaValidation:
    """Test JSON Schema validation functionality for Task 3.3."""
    
    def test_json_schema_validation_enabled(self):
        """Test that JSON Schema validation is enabled when schema is provided."""
        import tempfile
        import os
        
        # Create a temporary schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string", "format": "date-time"},
                        "language": {"type": "string", "default": "en"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            assert loader._schema is not None
            assert loader._schema["type"] == "object"
        finally:
            os.unlink(schema_path)
    
    def test_json_schema_validation_success(self):
        """Test successful JSON Schema validation."""
        import tempfile
        import os
        
        # Create a temporary schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            valid_data = {
                "metadata": {
                    "title": "Test Meeting",
                    "duration_seconds": 3.0,
                    "participant_count": 1,
                    "meeting_id": "meeting_123",
                    "date": "2023-01-01T10:00:00Z",
                    "language": "en"
                },
                "participants": [
                    {"id": "p1", "name": "Alice", "role": "participant"}
                ],
                "transcript": [
                    {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
                ]
            }
            
            # Should not raise exception
            assert loader.validate(valid_data) is True
        finally:
            os.unlink(schema_path)
    
    def test_json_schema_validation_failure_missing_required_field(self):
        """Test JSON Schema validation failure for missing required field."""
        import tempfile
        import os
        
        # Create a temporary schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            # Missing 'participants' field
            invalid_data = {
                "metadata": {
                    "title": "Test Meeting",
                    "duration_seconds": 3.0,
                    "participant_count": 1,
                    "meeting_id": "meeting_123",
                    "date": "2023-01-01T10:00:00Z",
                    "language": "en"
                },
                "transcript": [
                    {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
                ]
            }
            
            with pytest.raises(JSONSchemaError) as exc_info:
                loader.validate(invalid_data)
            assert "JSON schema validation failed" in str(exc_info.value)
        finally:
            os.unlink(schema_path)
    
    def test_json_schema_validation_failure_wrong_type(self):
        """Test JSON Schema validation failure for wrong data type."""
        import tempfile
        import os
        
        # Create a temporary schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            # Wrong type for duration_seconds (string instead of number)
            invalid_data = {
                "metadata": {
                    "title": "Test Meeting",
                    "duration_seconds": "not_a_number",  # Wrong type
                    "participant_count": 1,
                    "meeting_id": "meeting_123",
                    "date": "2023-01-01T10:00:00Z",
                    "language": "en"
                },
                "participants": [
                    {"id": "p1", "name": "Alice", "role": "participant"}
                ],
                "transcript": [
                    {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
                ]
            }
            
            with pytest.raises(JSONSchemaError) as exc_info:
                loader.validate(invalid_data)
            assert "JSON schema validation failed" in str(exc_info.value)
        finally:
            os.unlink(schema_path)
    
    def test_json_schema_validation_failure_constraint_violation(self):
        """Test JSON Schema validation failure for constraint violations."""
        import tempfile
        import os
        
        # Create a temporary schema file with strict constraints
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 10},  # Very short max length
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            # Title too long (exceeds maxLength: 10)
            invalid_data = {
                "metadata": {
                    "title": "This is a very long title that exceeds the maximum length",
                    "duration_seconds": 3.0,
                    "participant_count": 1,
                    "meeting_id": "meeting_123",
                    "date": "2023-01-01T10:00:00Z",
                    "language": "en"
                },
                "participants": [
                    {"id": "p1", "name": "Alice", "role": "participant"}
                ],
                "transcript": [
                    {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
                ]
            }
            
            with pytest.raises(JSONSchemaError) as exc_info:
                loader.validate(invalid_data)
            assert "JSON schema validation failed" in str(exc_info.value)
        finally:
            os.unlink(schema_path)
    
    def test_json_schema_validation_with_file_load(self):
        """Test JSON Schema validation during file loading."""
        import tempfile
        import os
        
        # Create a temporary schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as schema_file:
            json.dump(schema_content, schema_file)
            schema_path = schema_file.name
        
        # Create a temporary data file with invalid data
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": "not_a_number",  # Wrong type
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
            json.dump(invalid_data, data_file)
            data_path = data_file.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            with pytest.raises(JSONSchemaError) as exc_info:
                loader.load(data_path)
            assert "JSON schema validation failed" in str(exc_info.value)
        finally:
            os.unlink(schema_path)
            os.unlink(data_path)
    
    def test_json_schema_validation_error_messages(self):
        """Test that JSON Schema validation provides clear error messages."""
        import tempfile
        import os
        
        # Create a temporary schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            # Test various validation errors to ensure error messages are clear
            test_cases = [
                {
                    "description": "Missing required field",
                    "data": {
                        "metadata": {
                            "title": "Test Meeting",
                            "duration_seconds": 3.0,
                            "participant_count": 1,
                            "meeting_id": "meeting_123",
                            "date": "2023-01-01T10:00:00Z",
                            "language": "en"
                        },
                        "transcript": [
                            {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
                        ]
                        # Missing participants
                    }
                },
                {
                    "description": "Wrong data type",
                    "data": {
                        "metadata": {
                            "title": "Test Meeting",
                            "duration_seconds": 3.0,
                            "participant_count": "not_an_integer",  # Wrong type
                            "meeting_id": "meeting_123",
                            "date": "2023-01-01T10:00:00Z",
                            "language": "en"
                        },
                        "participants": [
                            {"id": "p1", "name": "Alice", "role": "participant"}
                        ],
                        "transcript": [
                            {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
                        ]
                    }
                },
                {
                    "description": "Constraint violation",
                    "data": {
                        "metadata": {
                            "title": "Test Meeting",
                            "duration_seconds": -1.0,  # Negative value
                            "participant_count": 1,
                            "meeting_id": "meeting_123",
                            "date": "2023-01-01T10:00:00Z",
                            "language": "en"
                        },
                        "participants": [
                            {"id": "p1", "name": "Alice", "role": "participant"}
                        ],
                        "transcript": [
                            {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
                        ]
                    }
                }
            ]
            
            for test_case in test_cases:
                with pytest.raises(JSONSchemaError) as exc_info:
                    loader.validate(test_case["data"])
                
                error_message = str(exc_info.value)
                assert "JSON schema validation failed" in error_message
                # Error message should contain information about the validation failure
                assert len(error_message) > 30  # Should be descriptive
        finally:
            os.unlink(schema_path)
    
    def test_json_schema_validation_without_schema(self):
        """Test that validation works without schema (Pydantic only)."""
        # Test with no schema path (should use default path or skip schema validation)
        loader = JSONTranscriptLoader()
        
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Should still work with just Pydantic validation
        assert loader.validate(valid_data) is True
    
    def test_json_schema_validation_with_additional_properties(self):
        """Test JSON Schema validation with additional properties."""
        import tempfile
        import os
        
        # Create a schema that allows additional properties
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "additionalProperties": True,
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["title", "duration_seconds", "participant_count", "meeting_id", "date"],
                    "additionalProperties": True,
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "duration_seconds": {"type": "number", "minimum": 0},
                        "participant_count": {"type": "integer", "minimum": 1},
                        "meeting_id": {"type": "string", "minLength": 1},
                        "date": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "additionalProperties": True,
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "role": {"type": "string"}
                        }
                    }
                },
                "transcript": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker_id", "start_time", "end_time", "text"],
                        "additionalProperties": True,
                        "properties": {
                            "speaker_id": {"type": "string", "minLength": 1},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "text": {"type": "string", "minLength": 1}
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            # Data with additional properties
            data_with_additional = {
                "metadata": {
                    "title": "Test Meeting",
                    "duration_seconds": 3.0,
                    "participant_count": 1,
                    "meeting_id": "meeting_123",
                    "date": "2023-01-01T10:00:00Z",
                    "language": "en",
                    "additional_metadata": "extra_value"  # Additional property
                },
                "participants": [
                    {"id": "p1", "name": "Alice", "role": "participant", "email": "alice@example.com"}  # Additional property
                ],
                "transcript": [
                    {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello", "confidence": 0.95}  # Additional property
                ],
                "additional_root_field": "extra_value"  # Additional property at root
            }
            
            # Should pass validation with additional properties allowed
            assert loader.validate(data_with_additional) is True
        finally:
            os.unlink(schema_path)


class TestAbstractionLayer:
    """Test abstraction layer for future format support (VTT, SRT) - Task 3.4."""
    
    def test_abstract_loader_interface_methods(self):
        """Test that AbstractTranscriptLoader defines required interface methods."""
        from abc import ABC
        from inspect import signature
        
        # Test that it's abstract
        assert issubclass(AbstractTranscriptLoader, ABC)
        
        # Test that it has required abstract methods
        abstract_methods = AbstractTranscriptLoader.__abstractmethods__
        expected_methods = {'load', 'validate', 'get_supported_formats'}
        
        assert expected_methods.issubset(abstract_methods)
        
        # Test method signatures
        assert hasattr(AbstractTranscriptLoader, 'load')
        assert hasattr(AbstractTranscriptLoader, 'validate')
        assert hasattr(AbstractTranscriptLoader, 'get_supported_formats')
    
    def test_abstract_loader_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        from inspect import signature
        
        # Test load method signature
        load_sig = signature(AbstractTranscriptLoader.load)
        assert 'file_path' in load_sig.parameters
        assert len(load_sig.parameters) == 2  # self and file_path
        
        # Test validate method signature
        validate_sig = signature(AbstractTranscriptLoader.validate)
        assert 'data' in validate_sig.parameters
        assert len(validate_sig.parameters) == 2  # self and data
        
        # Test get_supported_formats method signature
        formats_sig = signature(AbstractTranscriptLoader.get_supported_formats)
        assert len(formats_sig.parameters) == 1  # self only
    
    def test_abstract_loader_cannot_be_instantiated(self):
        """Test that AbstractTranscriptLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractTranscriptLoader()
    
    def test_factory_extensibility(self):
        """Test that TranscriptLoaderFactory can be extended with new formats."""
        # Test current supported formats
        formats = TranscriptLoaderFactory.get_supported_formats()
        assert '.json' in formats
        
        # Test that factory has internal loaders registry
        assert hasattr(TranscriptLoaderFactory, '_loaders')
        assert isinstance(TranscriptLoaderFactory._loaders, dict)
        assert '.json' in TranscriptLoaderFactory._loaders
    
    def test_factory_can_register_new_formats(self):
        """Test that factory can be extended to register new formats."""
        # Create a mock loader for testing
        class MockVTTLoader(AbstractTranscriptLoader):
            def load(self, file_path):
                return None
            
            def validate(self, data):
                return True
            
            def get_supported_formats(self):
                return ['.vtt']
        
        # Save original state
        original_loaders = TranscriptLoaderFactory._loaders.copy()
        
        try:
            # Register new format
            TranscriptLoaderFactory._loaders['.vtt'] = MockVTTLoader
            
            # Test that new format is supported
            formats = TranscriptLoaderFactory.get_supported_formats()
            assert '.vtt' in formats
            
            # Test that factory can create loader for new format
            loader = TranscriptLoaderFactory.create_loader("test.vtt")
            assert isinstance(loader, MockVTTLoader)
            
        finally:
            # Restore original state
            TranscriptLoaderFactory._loaders = original_loaders
    
    def test_factory_can_register_srt_format(self):
        """Test that factory can be extended to register SRT format."""
        # Create a mock SRT loader for testing
        class MockSRTLoader(AbstractTranscriptLoader):
            def load(self, file_path):
                return None
            
            def validate(self, data):
                return True
            
            def get_supported_formats(self):
                return ['.srt']
        
        # Save original state
        original_loaders = TranscriptLoaderFactory._loaders.copy()
        
        try:
            # Register new format
            TranscriptLoaderFactory._loaders['.srt'] = MockSRTLoader
            
            # Test that new format is supported
            formats = TranscriptLoaderFactory.get_supported_formats()
            assert '.srt' in formats
            
            # Test that factory can create loader for new format
            loader = TranscriptLoaderFactory.create_loader("test.srt")
            assert isinstance(loader, MockSRTLoader)
            
        finally:
            # Restore original state
            TranscriptLoaderFactory._loaders = original_loaders
    
    def test_multiple_format_registration(self):
        """Test that multiple formats can be registered simultaneously."""
        # Create mock loaders
        class MockVTTLoader(AbstractTranscriptLoader):
            def load(self, file_path):
                return None
            
            def validate(self, data):
                return True
            
            def get_supported_formats(self):
                return ['.vtt']
        
        class MockSRTLoader(AbstractTranscriptLoader):
            def load(self, file_path):
                return None
            
            def validate(self, data):
                return True
            
            def get_supported_formats(self):
                return ['.srt']
        
        # Save original state
        original_loaders = TranscriptLoaderFactory._loaders.copy()
        
        try:
            # Register multiple formats
            TranscriptLoaderFactory._loaders['.vtt'] = MockVTTLoader
            TranscriptLoaderFactory._loaders['.srt'] = MockSRTLoader
            
            # Test that all formats are supported
            formats = TranscriptLoaderFactory.get_supported_formats()
            assert '.json' in formats
            assert '.vtt' in formats
            assert '.srt' in formats
            
            # Test that factory can create loaders for all formats
            json_loader = TranscriptLoaderFactory.create_loader("test.json")
            vtt_loader = TranscriptLoaderFactory.create_loader("test.vtt")
            srt_loader = TranscriptLoaderFactory.create_loader("test.srt")
            
            assert isinstance(json_loader, JSONTranscriptLoader)
            assert isinstance(vtt_loader, MockVTTLoader)
            assert isinstance(srt_loader, MockSRTLoader)
            
        finally:
            # Restore original state
            TranscriptLoaderFactory._loaders = original_loaders
    
    def test_interface_consistency_across_implementations(self):
        """Test that all implementations follow the same interface."""
        # Test JSON loader implements interface correctly
        json_loader = JSONTranscriptLoader()
        
        # Test that all required methods exist
        assert hasattr(json_loader, 'load')
        assert hasattr(json_loader, 'validate')
        assert hasattr(json_loader, 'get_supported_formats')
        
        # Test that methods are callable
        assert callable(json_loader.load)
        assert callable(json_loader.validate)
        assert callable(json_loader.get_supported_formats)
        
        # Test that get_supported_formats returns correct type
        formats = json_loader.get_supported_formats()
        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)
    
    def test_factory_handles_case_insensitive_extensions(self):
        """Test that factory handles case-insensitive file extensions."""
        # Test that both .json and .JSON work
        loader_lower = TranscriptLoaderFactory.create_loader("test.json")
        loader_upper = TranscriptLoaderFactory.create_loader("test.JSON")
        
        assert isinstance(loader_lower, JSONTranscriptLoader)
        assert isinstance(loader_upper, JSONTranscriptLoader)
    
    def test_factory_error_handling_for_unsupported_formats(self):
        """Test that factory provides clear error messages for unsupported formats."""
        unsupported_formats = ['.txt', '.xml', '.csv', '.unknown']
        
        for fmt in unsupported_formats:
            with pytest.raises(TranscriptLoaderError) as exc_info:
                TranscriptLoaderFactory.create_loader(f"test{fmt}")
            
            error_msg = str(exc_info.value)
            assert "Unsupported file format" in error_msg
            assert fmt in error_msg
            assert "Supported formats" in error_msg
    
    def test_factory_with_path_object(self):
        """Test that factory works with Path objects."""
        from pathlib import Path
        
        # Test with Path object
        path_obj = Path("test.json")
        loader = TranscriptLoaderFactory.create_loader(path_obj)
        
        assert isinstance(loader, JSONTranscriptLoader)
    
    def test_factory_with_complex_file_paths(self):
        """Test that factory works with complex file paths."""
        test_paths = [
            "/path/to/file.json",
            "relative/path/file.json",
            "file.with.dots.json",
            "/very/long/path/to/some/deeply/nested/file.json"
        ]
        
        for path in test_paths:
            loader = TranscriptLoaderFactory.create_loader(path)
            assert isinstance(loader, JSONTranscriptLoader)
    
    def test_factory_preserves_kwargs(self):
        """Test that factory passes kwargs to loader constructors."""
        # Test with schema_path kwarg
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"type": "object"}')):
                loader = TranscriptLoaderFactory.create_loader(
                    "test.json", 
                    schema_path="custom_schema.json"
                )
                
                assert isinstance(loader, JSONTranscriptLoader)
                assert loader.schema_path == "custom_schema.json"
    
    def test_abstract_loader_documentation(self):
        """Test that AbstractTranscriptLoader has proper documentation."""
        assert AbstractTranscriptLoader.__doc__ is not None
        assert "Abstract base class" in AbstractTranscriptLoader.__doc__
        
        # Test that abstract methods have documentation
        assert AbstractTranscriptLoader.load.__doc__ is not None
        assert AbstractTranscriptLoader.validate.__doc__ is not None
        assert AbstractTranscriptLoader.get_supported_formats.__doc__ is not None
    
    def test_extensibility_pattern_demonstration(self):
        """Test demonstration of how to extend the system for new formats."""
        # This test demonstrates the extensibility pattern for future formats
        
        # Step 1: Create a new loader class
        class VTTTranscriptLoader(AbstractTranscriptLoader):
            """VTT transcript loader implementation."""
            
            def load(self, file_path):
                """Load VTT transcript data from file."""
                # Mock implementation - in real use, this would parse VTT format
                return TranscriptDataModel(
                    metadata=MetadataModel(
                        title="VTT Test",
                        duration_seconds=10.0,
                        participant_count=1,
                        meeting_id="vtt_test",
                        date="2023-01-01T10:00:00Z"
                    ),
                    participants=[
                        ParticipantModel(id="p1", name="VTT Speaker", role="speaker")
                    ],
                    transcript=[
                        TranscriptEntryModel(
                            speaker_id="p1",
                            start_time=0.0,
                            end_time=5.0,
                            text="VTT transcript text"
                        )
                    ]
                )
            
            def validate(self, data):
                """Validate VTT transcript data."""
                # Mock validation - in real use, this would validate VTT-specific format
                return True
            
            def get_supported_formats(self):
                """Get supported formats."""
                return ['.vtt']
        
        # Step 2: Register the new loader
        original_loaders = TranscriptLoaderFactory._loaders.copy()
        
        try:
            TranscriptLoaderFactory._loaders['.vtt'] = VTTTranscriptLoader
            
            # Step 3: Test that the new format works
            loader = TranscriptLoaderFactory.create_loader("test.vtt")
            assert isinstance(loader, VTTTranscriptLoader)
            
            # Step 4: Test that load_transcript convenience function works
            with patch.object(VTTTranscriptLoader, 'load') as mock_load:
                mock_load.return_value = TranscriptDataModel(
                    metadata=MetadataModel(
                        title="VTT Test",
                        duration_seconds=10.0,
                        participant_count=1,
                        meeting_id="vtt_test",
                        date="2023-01-01T10:00:00Z"
                    ),
                    participants=[
                        ParticipantModel(id="p1", name="VTT Speaker", role="speaker")
                    ],
                    transcript=[
                        TranscriptEntryModel(
                            speaker_id="p1",
                            start_time=0.0,
                            end_time=5.0,
                            text="VTT transcript text"
                        )
                    ]
                )
                
                result = load_transcript("test.vtt")
                assert isinstance(result, TranscriptDataModel)
                assert result.metadata.title == "VTT Test"
                mock_load.assert_called_once_with("test.vtt")
            
        finally:
            TranscriptLoaderFactory._loaders = original_loaders
    
    def test_interface_design_for_format_specific_features(self):
        """Test that interface design supports format-specific features."""
        # Test that loaders can have format-specific initialization parameters
        
        class ExtendedVTTLoader(AbstractTranscriptLoader):
            def __init__(self, encoding='utf-8', strict_timing=True):
                self.encoding = encoding
                self.strict_timing = strict_timing
            
            def load(self, file_path):
                return None
            
            def validate(self, data):
                return True
            
            def get_supported_formats(self):
                return ['.vtt']
        
        # Test that factory can pass format-specific parameters
        original_loaders = TranscriptLoaderFactory._loaders.copy()
        
        try:
            TranscriptLoaderFactory._loaders['.vtt'] = ExtendedVTTLoader
            
            # Test with format-specific parameters
            loader = TranscriptLoaderFactory.create_loader(
                "test.vtt",
                encoding='utf-16',
                strict_timing=False
            )
            
            assert isinstance(loader, ExtendedVTTLoader)
            assert loader.encoding == 'utf-16'
            assert loader.strict_timing is False
            
        finally:
            TranscriptLoaderFactory._loaders = original_loaders
    
    def test_abstract_loader_type_annotations(self):
        """Test that abstract loader has proper type annotations."""
        from typing import get_type_hints
        
        # Test load method type hints
        load_hints = get_type_hints(AbstractTranscriptLoader.load)
        assert 'file_path' in load_hints
        assert 'return' in load_hints
        
        # Test validate method type hints
        validate_hints = get_type_hints(AbstractTranscriptLoader.validate)
        assert 'data' in validate_hints
        assert 'return' in validate_hints
        
        # Test get_supported_formats method type hints
        formats_hints = get_type_hints(AbstractTranscriptLoader.get_supported_formats)
        assert 'return' in formats_hints
    
    def test_mock_implementation_for_testing(self):
        """Test that mock implementations can be created for testing."""
        # This demonstrates how to create mock implementations for testing
        
        class MockTranscriptLoader(AbstractTranscriptLoader):
            def __init__(self, mock_data=None, should_fail=False):
                self.mock_data = mock_data
                self.should_fail = should_fail
            
            def load(self, file_path):
                if self.should_fail:
                    raise ValidationError("Mock validation error")
                return self.mock_data
            
            def validate(self, data):
                if self.should_fail:
                    raise ValidationError("Mock validation error")
                return True
            
            def get_supported_formats(self):
                return ['.mock']
        
        # Test normal operation
        mock_data = TranscriptDataModel(
            metadata=MetadataModel(
                title="Mock Test",
                duration_seconds=5.0,
                participant_count=1,
                meeting_id="mock_test",
                date="2023-01-01T10:00:00Z"
            ),
            participants=[
                ParticipantModel(id="p1", name="Mock Speaker", role="speaker")
            ],
            transcript=[
                TranscriptEntryModel(
                    speaker_id="p1",
                    start_time=0.0,
                    end_time=5.0,
                    text="Mock transcript text"
                )
            ]
        )
        
        loader = MockTranscriptLoader(mock_data=mock_data)
        
        assert loader.load("test.mock") == mock_data
        assert loader.validate({}) is True
        assert loader.get_supported_formats() == ['.mock']
        
        # Test error handling
        failing_loader = MockTranscriptLoader(should_fail=True)
        
        with pytest.raises(ValidationError):
            failing_loader.load("test.mock")
        
        with pytest.raises(ValidationError):
            failing_loader.validate({})


class TestFileLoadingErrorHandling:
    """Test comprehensive file loading error handling for Task 3.5."""
    
    def test_loader_initialization_with_max_file_size(self):
        """Test that loader can be initialized with custom max file size."""
        loader = JSONTranscriptLoader(max_file_size=1024)
        assert loader.max_file_size == 1024
        
        loader_default = JSONTranscriptLoader()
        assert loader_default.max_file_size == JSONTranscriptLoader.DEFAULT_MAX_FILE_SIZE
    
    def test_file_not_found_error(self):
        """Test FileNotFoundError for non-existent files."""
        loader = JSONTranscriptLoader()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("non_existent_file.json")
        
        error_msg = str(exc_info.value)
        assert "Transcript file not found" in error_msg
        assert "non_existent_file.json" in error_msg
        assert "Please check the file path" in error_msg
    
    def test_file_not_found_error_for_directory(self):
        """Test FileNotFoundError when path is a directory."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(FileNotFoundError) as exc_info:
                loader.load(temp_dir)
            
            error_msg = str(exc_info.value)
            assert "Path exists but is not a file" in error_msg
            assert "Please provide a path to a valid JSON file" in error_msg
    
    @patch('os.access', return_value=False)
    def test_file_permission_error(self, mock_access):
        """Test FilePermissionError for files without read permission."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(FilePermissionError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Permission denied" in error_msg
            assert "Cannot read file" in error_msg
            assert "Please check file permissions" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_file_size_error_large_file(self):
        """Test FileSizeError for files exceeding size limit."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write a small file but set a very small size limit
            f.write('{"test": "data"}')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader(max_file_size=5)  # Very small limit
            
            with pytest.raises(FileSizeError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "File size" in error_msg
            assert "exceeds maximum allowed size" in error_msg
            assert "Consider splitting the file" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_file_corruption_error_empty_file(self):
        """Test FileCorruptionError for empty files."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create empty file
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(FileCorruptionError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "File is empty" in error_msg
            assert "Please provide a valid JSON file" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_file_corruption_error_null_bytes(self):
        """Test FileCorruptionError for files with null bytes."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as f:
            # Write file with null bytes
            f.write(b'{"test": "data\x00"}')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(FileCorruptionError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "File contains null bytes" in error_msg
            assert "may be corrupted" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_file_encoding_error_unsupported_encoding(self):
        """Test FileEncodingError for unsupported encodings."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as f:
            # Write file with unsupported encoding
            f.write(b'\xff\xfe{"test": "data"}')  # Invalid UTF-8
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(FileEncodingError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Could not decode file" in error_msg
            assert "supported encodings" in error_msg
            assert "Please ensure the file is properly encoded" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_validation_error_invalid_json(self):
        """Test ValidationError for invalid JSON syntax."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write invalid JSON
            f.write('{"test": "data",}')  # Trailing comma
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Invalid JSON" in error_msg
            assert "Please check the JSON syntax" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_validation_error_non_object_json(self):
        """Test ValidationError for JSON that isn't an object."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write valid JSON but not an object
            f.write('["not", "an", "object"]')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "must contain a JSON object at the root level" in error_msg
            assert "but found list" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_successful_load_with_valid_data(self):
        """Test successful file loading with valid data."""
        import tempfile
        
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_data, f)
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            result = loader.load(temp_path)
            
            assert isinstance(result, TranscriptDataModel)
            assert result.metadata.title == "Test Meeting"
            assert len(result.participants) == 1
            assert len(result.transcript) == 1
        finally:
            os.unlink(temp_path)
    
    def test_successful_load_with_different_encodings(self):
        """Test successful file loading with different encodings."""
        import tempfile
        
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Test with UTF-8 with BOM
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8-sig') as f:
            json.dump(valid_data, f)
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            result = loader.load(temp_path)
            
            assert isinstance(result, TranscriptDataModel)
            assert result.metadata.title == "Test Meeting"
        finally:
            os.unlink(temp_path)
    
    def test_integrity_check_integration(self):
        """Test that integrity checks are properly integrated."""
        import tempfile
        
        # Test with invalid speaker ID
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p2", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}  # Invalid speaker
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Invalid speaker IDs" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_pydantic_validation_integration(self):
        """Test that Pydantic validation is properly integrated."""
        import tempfile
        
        # Test with missing required field
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "role": "participant"}  # Missing name
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Data validation failed" in error_msg
            assert "Please check that all required fields are present" in error_msg
        finally:
            os.unlink(temp_path)
    
    @patch('os.access', side_effect=OSError("Permission check failed"))
    def test_file_permission_error_os_error(self, mock_access):
        """Test FilePermissionError when os.access raises OSError."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(FilePermissionError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Error checking file permissions" in error_msg
        finally:
            os.unlink(temp_path)
    
    @patch('pathlib.Path.stat', side_effect=OSError("Stat failed"))
    def test_file_corruption_error_stat_failure(self, mock_stat):
        """Test FileCorruptionError when file stat fails."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            with pytest.raises(FileCorruptionError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Error accessing file" in error_msg
            assert "File may be corrupted or inaccessible" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_file_extension_flexibility(self):
        """Test that loader is flexible with file extensions."""
        import tempfile
        
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        # Test with .txt extension (should still work)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            json.dump(valid_data, f)
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            result = loader.load(temp_path)
            
            assert isinstance(result, TranscriptDataModel)
            assert result.metadata.title == "Test Meeting"
        finally:
            os.unlink(temp_path)
    
    def test_memory_error_simulation(self):
        """Test FileSizeError when MemoryError occurs."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            # Mock json.loads to raise MemoryError
            with patch('json.loads', side_effect=MemoryError("Out of memory")):
                with pytest.raises(FileSizeError) as exc_info:
                    loader.load(temp_path)
                
                error_msg = str(exc_info.value)
                assert "Not enough memory to load file" in error_msg
                assert "File may be too large" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_json_schema_validation_error_handling(self):
        """Test JSON schema validation error handling."""
        import tempfile
        
        # Create a simple schema
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["required_field"],
            "properties": {
                "required_field": {"type": "string"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as schema_file:
            json.dump(schema_content, schema_file)
            schema_path = schema_file.name
        
        # Create invalid data
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
            json.dump(invalid_data, data_file)
            data_path = data_file.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            
            with pytest.raises(JSONSchemaError) as exc_info:
                loader.load(data_path)
            
            error_msg = str(exc_info.value)
            assert "JSON schema validation failed" in error_msg
            assert "Please ensure the file structure matches" in error_msg
        finally:
            os.unlink(schema_path)
            os.unlink(data_path)
    
    def test_unexpected_exception_handling(self):
        """Test handling of unexpected exceptions."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name
        
        try:
            loader = JSONTranscriptLoader()
            
            # Mock open to raise an unexpected exception
            with patch('builtins.open', side_effect=RuntimeError("Unexpected error")):
                with pytest.raises(TranscriptLoaderError) as exc_info:
                    loader.load(temp_path)
                
                error_msg = str(exc_info.value)
                assert "Unexpected error while reading file" in error_msg
                assert "Please check the file and try again" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_error_message_quality(self):
        """Test that error messages are clear and actionable."""
        loader = JSONTranscriptLoader()
        
        # Test FileNotFoundError message quality
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("missing_file.json")
        
        error_msg = str(exc_info.value)
        assert "Transcript file not found" in error_msg
        assert "missing_file.json" in error_msg
        assert "Please check the file path" in error_msg
        assert "ensure the file exists" in error_msg
        
        # Error message should be descriptive and actionable
        assert len(error_msg) > 50  # Should be reasonably detailed

    def test_comprehensive_error_message_clarity(self):
        """Test that all error messages are clear, actionable, and provide context."""
        import tempfile
        import os
        
        loader = JSONTranscriptLoader()
        
        # Test 1: File not found error
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("nonexistent_file.json")
        
        error_msg = str(exc_info.value)
        assert "Transcript file not found" in error_msg
        assert "nonexistent_file.json" in error_msg
        assert "Please check the file path" in error_msg
        assert "ensure the file exists" in error_msg
        
        # Test 2: Directory instead of file error
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError) as exc_info:
                loader.load(temp_dir)
            
            error_msg = str(exc_info.value)
            assert "Path exists but is not a file" in error_msg
            assert "Please provide a path to a valid JSON file" in error_msg
        
        # Test 3: File permission error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write('{"test": "data"}')
            temp_path = temp_file.name
        
        try:
            # Remove read permissions
            os.chmod(temp_path, 0o000)
            
            with pytest.raises(FilePermissionError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Permission denied" in error_msg
            assert "Cannot read file" in error_msg
            assert "Please check file permissions" in error_msg
            assert "ensure read access" in error_msg
            
        finally:
            # Restore permissions to delete file
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)
        
        # Test 4: Empty file error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            # File is empty
        
        try:
            with pytest.raises(FileCorruptionError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "File is empty" in error_msg
            assert "Please provide a valid JSON file with transcript data" in error_msg
            
        finally:
            os.unlink(temp_path)
        
        # Test 5: File too large error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write('{"test": "data"}')
        
        try:
            # Create loader with very small max size
            small_loader = JSONTranscriptLoader(max_file_size=5)
            
            with pytest.raises(FileSizeError) as exc_info:
                small_loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "File size" in error_msg
            assert "exceeds maximum allowed size" in error_msg
            assert "Consider splitting the file or increasing the size limit" in error_msg
            
        finally:
            os.unlink(temp_path)
        
        # Test 6: Invalid JSON syntax error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write('{"invalid": json}')  # Missing quotes around json
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Invalid JSON in file" in error_msg
            assert "Please check the JSON syntax" in error_msg
            assert "ensure the file is properly formatted" in error_msg
            assert "line" in error_msg  # Should include line number
            
        finally:
            os.unlink(temp_path)
        
        # Test 7: Wrong data type at root level
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write('["not", "an", "object"]')  # Array instead of object
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "must contain a JSON object at the root level" in error_msg
            assert "Please ensure the file contains a valid transcript object" in error_msg
            
        finally:
            os.unlink(temp_path)
        
        # Test 8: Missing required keys (this will likely trigger JSON schema validation)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write('{"metadata": {}}')  # Missing participants and transcript
            temp_path = temp_file.name
        
        try:
            with pytest.raises((ValidationError, JSONSchemaError)) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            # Could be either missing required keys or JSON schema validation error
            assert ("Missing required keys" in error_msg or 
                    "JSON schema validation failed" in error_msg or
                    "Please ensure the file structure matches" in error_msg)
            
        finally:
            os.unlink(temp_path)
        
        # Test 9: Invalid speaker ID consistency
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p2", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}  # p2 not in participants
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(invalid_data, temp_file)
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Invalid speaker IDs in transcript" in error_msg
            assert "All speaker IDs must be defined in participants list" in error_msg
            
        finally:
            os.unlink(temp_path)
        
        # Test 10: Timestamp ordering error
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 10.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 5.0, "end_time": 7.0, "text": "Second"},
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "First"}  # Out of order
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(invalid_data, temp_file)
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Transcript entries must be ordered by start_time" in error_msg
            
        finally:
            os.unlink(temp_path)
    
    def test_error_message_actionability(self):
        """Test that error messages provide specific actionable guidance."""
        import tempfile
        import os
        
        loader = JSONTranscriptLoader()
        
        # Test 1: Pydantic validation error with clear field reference
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": -5.0,  # Invalid: negative duration
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(invalid_data, temp_file)
            temp_path = temp_file.name
        
        try:
            with pytest.raises((ValidationError, JSONSchemaError)) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            # Could be either data validation failed or JSON schema validation error
            assert ("Data validation failed" in error_msg or 
                    "JSON schema validation failed" in error_msg)
            assert ("Please check that all required fields are present" in error_msg or
                    "Please ensure the file structure matches" in error_msg)
            
        finally:
            os.unlink(temp_path)
        
        # Test 2: Duration consistency error with specific values
        invalid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 100.0,  # Says 100 seconds
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 10.0, "text": "Hello"}  # Actually ends at 10
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(invalid_data, temp_file)
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Duration mismatch" in error_msg
            assert "metadata says 100.0s" in error_msg
            assert "transcript ends at 10.0s" in error_msg
            
        finally:
            os.unlink(temp_path)
        
        # Test 3: File encoding error with supported encodings list
        # Create a file with invalid encoding (binary content)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as temp_file:
            temp_file.write(b'\x80\x81\x82\x83')  # Invalid UTF-8 bytes
            temp_path = temp_file.name
        
        try:
            with pytest.raises(FileEncodingError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "Could not decode file" in error_msg
            assert "supported encodings" in error_msg
            assert "utf-8" in error_msg
            assert "Please ensure the file is properly encoded" in error_msg
            
        finally:
            os.unlink(temp_path)
    
    def test_error_message_context_information(self):
        """Test that error messages include sufficient context for debugging."""
        import tempfile
        import os
        
        loader = JSONTranscriptLoader()
        
        # Test error messages include file path context
        test_file_path = "/path/to/test/file.json"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load(test_file_path)
        
        error_msg = str(exc_info.value)
        assert test_file_path in error_msg
        
        # Test JSON syntax error includes line and column information
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write('{\n  "valid": "json",\n  "invalid": syntax\n}')  # Missing quotes
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            assert "line" in error_msg
            assert "column" in error_msg
            assert temp_path in error_msg
            
        finally:
            os.unlink(temp_path)
        
        # Test validation error includes specific validation details
        invalid_data = {
            "metadata": {
                "title": "",  # Empty title should fail validation
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(invalid_data, temp_file)
            temp_path = temp_file.name
        
        try:
            with pytest.raises((ValidationError, JSONSchemaError)) as exc_info:
                loader.load(temp_path)
            
            error_msg = str(exc_info.value)
            # Could be either data validation failed or JSON schema validation error
            assert ("Data validation failed" in error_msg or 
                    "JSON schema validation failed" in error_msg)
            assert temp_path in error_msg
            
        finally:
            os.unlink(temp_path)
    
    def test_file_loading_with_schema_validation(self):
        """Test file loading with schema validation enabled."""
        import tempfile
        
        # Create a schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["metadata", "participants", "transcript"],
            "properties": {
                "metadata": {"type": "object"},
                "participants": {"type": "array"},
                "transcript": {"type": "array"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as schema_file:
            json.dump(schema_content, schema_file)
            schema_path = schema_file.name
        
        # Create valid data
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
            json.dump(valid_data, data_file)
            data_path = data_file.name
        
        try:
            loader = JSONTranscriptLoader(schema_path=schema_path)
            result = loader.load(data_path)
            
            assert isinstance(result, TranscriptDataModel)
            assert result.metadata.title == "Test Meeting"
        finally:
            os.unlink(schema_path)
            os.unlink(data_path)
    
    def test_concurrent_file_loading(self):
        """Test that file loading works with concurrent access."""
        import tempfile
        import threading
        
        valid_data = {
            "metadata": {
                "title": "Test Meeting",
                "duration_seconds": 3.0,
                "participant_count": 1,
                "meeting_id": "meeting_123",
                "date": "2023-01-01T10:00:00Z",
                "language": "en"
            },
            "participants": [
                {"id": "p1", "name": "Alice", "role": "participant"}
            ],
            "transcript": [
                {"speaker_id": "p1", "start_time": 0.0, "end_time": 3.0, "text": "Hello"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_data, f)
            temp_path = f.name
        
        results = []
        errors = []
        
        def load_file():
            try:
                loader = JSONTranscriptLoader()
                result = loader.load(temp_path)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        try:
            # Start multiple threads to load the same file
            threads = []
            for i in range(5):
                thread = threading.Thread(target=load_file)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 5
            
            # All results should be identical
            for result in results:
                assert isinstance(result, TranscriptDataModel)
                assert result.metadata.title == "Test Meeting"
                
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])