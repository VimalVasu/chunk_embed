"""
Test malformed data validation for Task 2.5
Validates malformed JSON samples for error testing
"""
import json
import pytest
from pathlib import Path


class TestMalformedData:
    """Test that malformed data files serve their purpose for error testing"""
    
    @pytest.fixture
    def malformed_dir(self):
        """Get malformed directory"""
        return Path(__file__).parent.parent / "test_data" / "malformed"
    
    @pytest.fixture
    def malformed_files(self, malformed_dir):
        """Get all malformed test files"""
        return list(malformed_dir.glob("*.json"))
    
    def test_malformed_directory_has_files(self, malformed_files):
        """Test that malformed directory contains test files"""
        assert len(malformed_files) > 0, "malformed/ directory must contain at least one test file"
    
    def test_invalid_json_syntax_file_exists(self, malformed_files):
        """Test that we have a file with invalid JSON syntax"""
        syntax_files = [f for f in malformed_files if "syntax" in f.name]
        assert len(syntax_files) > 0, "Must have at least one invalid JSON syntax file"
        
        # Verify the syntax file actually has invalid JSON
        for file_path in syntax_files:
            with open(file_path, 'r') as f:
                content = f.read()
            
            with pytest.raises(json.JSONDecodeError):
                json.loads(content)
    
    def test_missing_required_fields_file_exists(self, malformed_files):
        """Test that we have a file missing required fields"""
        missing_files = [f for f in malformed_files if "missing" in f.name or "required" in f.name]
        assert len(missing_files) > 0, "Must have at least one file missing required fields"
        
        # Verify the file is valid JSON but missing required fields
        for file_path in missing_files:
            with open(file_path, 'r') as f:
                data = json.load(f)  # Should not raise JSONDecodeError
            
            # Should be missing at least one required top-level field
            required_keys = {"metadata", "participants", "transcript"}
            present_keys = set(data.keys())
            missing_keys = required_keys - present_keys
            assert len(missing_keys) > 0, f"{file_path.name} should be missing some required fields"
    
    def test_invalid_data_types_file_exists(self, malformed_files):
        """Test that we have a file with invalid data types"""
        type_files = [f for f in malformed_files if "type" in f.name or "data_type" in f.name]
        assert len(type_files) > 0, "Must have at least one file with invalid data types"
        
        # Verify the file has wrong data types
        for file_path in type_files:
            with open(file_path, 'r') as f:
                data = json.load(f)  # Should not raise JSONDecodeError
            
            # Check for common type violations
            if "metadata" in data:
                metadata = data["metadata"]
                type_errors_found = False
                
                # duration_seconds should be numeric but might be string
                if "duration_seconds" in metadata and isinstance(metadata["duration_seconds"], str):
                    type_errors_found = True
                
                # participant_count should be int but might be string
                if "participant_count" in metadata and isinstance(metadata["participant_count"], str):
                    type_errors_found = True
                
                # meeting_id should be string but might be numeric
                if "meeting_id" in metadata and isinstance(metadata["meeting_id"], (int, float)):
                    type_errors_found = True
                
                assert type_errors_found, f"{file_path.name} should have data type violations"
    
    def test_empty_content_file_exists(self, malformed_files):
        """Test that we have a file with empty content"""
        empty_files = [f for f in malformed_files if "empty" in f.name]
        assert len(empty_files) > 0, "Must have at least one file with empty content"
        
        # Verify the file has empty arrays or zero values
        for file_path in empty_files:
            with open(file_path, 'r') as f:
                data = json.load(f)  # Should not raise JSONDecodeError
            
            empty_conditions = []
            if "participants" in data:
                empty_conditions.append(len(data["participants"]) == 0)
            if "transcript" in data:
                empty_conditions.append(len(data["transcript"]) == 0)
            if "metadata" in data and "participant_count" in data["metadata"]:
                empty_conditions.append(data["metadata"]["participant_count"] == 0)
            
            assert any(empty_conditions), f"{file_path.name} should have empty content"
    
    def test_inconsistent_speakers_file_exists(self, malformed_files):
        """Test that we have a file with inconsistent speaker references"""
        speaker_files = [f for f in malformed_files if "speaker" in f.name or "inconsistent" in f.name]
        assert len(speaker_files) > 0, "Must have at least one file with inconsistent speakers"
        
        # Verify speaker inconsistencies
        for file_path in speaker_files:
            with open(file_path, 'r') as f:
                data = json.load(f)  # Should not raise JSONDecodeError
            
            if "participants" in data and "transcript" in data:
                participant_ids = {p["id"] for p in data["participants"]}
                transcript_speaker_ids = {entry["speaker_id"] for entry in data["transcript"]}
                
                # Should have speaker_ids in transcript that don't exist in participants
                invalid_speakers = transcript_speaker_ids - participant_ids
                assert len(invalid_speakers) > 0, f"{file_path.name} should have inconsistent speaker references"
    
    def test_invalid_timestamps_file_exists(self, malformed_files):
        """Test that we have a file with invalid timestamps"""
        timestamp_files = [f for f in malformed_files if "timestamp" in f.name or "time" in f.name]
        assert len(timestamp_files) > 0, "Must have at least one file with invalid timestamps"
        
        # Verify timestamp violations
        for file_path in timestamp_files:
            with open(file_path, 'r') as f:
                data = json.load(f)  # Should not raise JSONDecodeError
            
            if "transcript" in data:
                timestamp_errors_found = False
                
                for entry in data["transcript"]:
                    if "start_time" in entry and "end_time" in entry:
                        start_time = entry["start_time"]
                        end_time = entry["end_time"]
                        
                        # Check for negative start times
                        if start_time < 0:
                            timestamp_errors_found = True
                        
                        # Check for end_time <= start_time
                        if end_time <= start_time:
                            timestamp_errors_found = True
                
                assert timestamp_errors_found, f"{file_path.name} should have timestamp violations"
    
    def test_malformed_data_variety(self, malformed_files):
        """Test that we have variety in malformed data types"""
        # Should have at least 5 different types of malformed data
        assert len(malformed_files) >= 5, "Must have at least 5 different malformed data files"
        
        # Check that we have diverse error types based on filenames
        error_types = set()
        for file_path in malformed_files:
            filename = file_path.name.lower()
            if "syntax" in filename:
                error_types.add("syntax")
            elif "missing" in filename or "required" in filename:
                error_types.add("missing_fields")
            elif "type" in filename:
                error_types.add("data_types")
            elif "empty" in filename:
                error_types.add("empty_content")
            elif "speaker" in filename or "inconsistent" in filename:
                error_types.add("speaker_inconsistency")
            elif "timestamp" in filename or "time" in filename:
                error_types.add("invalid_timestamps")
            else:
                error_types.add("other")
        
        assert len(error_types) >= 4, f"Must have variety in error types. Found: {error_types}"
    
    def test_malformed_files_serve_testing_purpose(self, malformed_files):
        """Test that malformed files are actually useful for error testing"""
        valid_json_count = 0
        error_prone_count = 0
        
        for file_path in malformed_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                valid_json_count += 1
                
                # Even if JSON is valid, it should have logical errors for testing
                has_logical_errors = False
                
                # Check for various logical errors
                if "participants" in data and len(data["participants"]) == 0:
                    has_logical_errors = True
                elif "transcript" in data and len(data["transcript"]) == 0:
                    has_logical_errors = True
                elif "participants" in data and "transcript" in data:
                    # Check speaker consistency
                    participant_ids = {p["id"] for p in data["participants"]}
                    transcript_speaker_ids = {entry["speaker_id"] for entry in data["transcript"]}
                    if transcript_speaker_ids - participant_ids:
                        has_logical_errors = True
                
                # Check for missing required top-level keys
                required_keys = {"metadata", "participants", "transcript"}
                if not required_keys.issubset(data.keys()):
                    has_logical_errors = True
                
                # Check for data type errors
                if "metadata" in data:
                    metadata = data["metadata"]
                    if "duration_seconds" in metadata and isinstance(metadata["duration_seconds"], str):
                        has_logical_errors = True
                    if "participant_count" in metadata and isinstance(metadata["participant_count"], str):
                        has_logical_errors = True
                
                # Check for timestamp errors
                if "transcript" in data:
                    for entry in data["transcript"]:
                        if "start_time" in entry and "end_time" in entry:
                            try:
                                start_time = entry["start_time"]
                                end_time = entry["end_time"]
                                if start_time < 0 or end_time <= start_time:
                                    has_logical_errors = True
                            except (TypeError, ValueError):
                                has_logical_errors = True
                
                if has_logical_errors:
                    error_prone_count += 1
                    
            except json.JSONDecodeError:
                error_prone_count += 1
        
        # All malformed files should either have JSON syntax errors or logical errors
        assert error_prone_count == len(malformed_files), \
            f"All malformed files should have errors. Found {error_prone_count}/{len(malformed_files)} with errors"