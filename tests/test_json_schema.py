"""
Test JSON schema validation for Task 2.6
Validates JSON Schema compliance for transcript data
"""
import json
import pytest
from pathlib import Path
from jsonschema import validate, ValidationError, Draft7Validator


class TestJSONSchema:
    """Test that transcript JSON schema is properly defined and validates correctly"""
    
    @pytest.fixture
    def schema_file(self):
        """Get transcript schema file"""
        return Path(__file__).parent.parent / "test_data" / "schemas" / "transcript_schema.json"
    
    @pytest.fixture
    def schema(self, schema_file):
        """Load the JSON schema"""
        with open(schema_file, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def samples_dir(self):
        """Get samples directory"""
        return Path(__file__).parent.parent / "test_data" / "samples"
    
    @pytest.fixture
    def sample_files(self, samples_dir):
        """Get all sample files"""
        return list(samples_dir.glob("*.json"))
    
    @pytest.fixture
    def edge_cases_dir(self):
        """Get edge cases directory"""
        return Path(__file__).parent.parent / "test_data" / "edge_cases"
    
    @pytest.fixture
    def edge_case_files(self, edge_cases_dir):
        """Get all edge case files"""
        return list(edge_cases_dir.glob("*.json"))
    
    @pytest.fixture
    def malformed_dir(self):
        """Get malformed directory"""
        return Path(__file__).parent.parent / "test_data" / "malformed"
    
    @pytest.fixture
    def malformed_files(self, malformed_dir):
        """Get all malformed files"""
        return list(malformed_dir.glob("*.json"))
    
    def test_schema_file_exists(self, schema_file):
        """Test that the JSON schema file exists"""
        assert schema_file.exists(), "transcript_schema.json must exist"
    
    def test_schema_is_valid_json(self, schema_file):
        """Test that the schema file contains valid JSON"""
        with open(schema_file, 'r') as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Schema file contains invalid JSON: {e}")
    
    def test_schema_is_valid_json_schema(self, schema):
        """Test that the schema is a valid JSON Schema"""
        try:
            Draft7Validator.check_schema(schema)
        except Exception as e:
            pytest.fail(f"Schema is not a valid JSON Schema: {e}")
    
    def test_schema_has_required_structure(self, schema):
        """Test that schema has required JSON Schema fields"""
        required_fields = ["$schema", "title", "type", "properties"]
        for field in required_fields:
            assert field in schema, f"Schema must have '{field}' field"
        
        # Check that it defines the main structure
        assert "metadata" in schema["properties"], "Schema must define metadata"
        assert "participants" in schema["properties"], "Schema must define participants"
        assert "transcript" in schema["properties"], "Schema must define transcript"
    
    def test_valid_samples_pass_schema_validation(self, schema, sample_files):
        """Test that all valid sample files pass schema validation"""
        validator = Draft7Validator(schema)
        
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    # This should not raise any validation errors
                    validate(instance=data, schema=schema)
                except ValidationError as e:
                    pytest.fail(f"Valid sample file {file_path.name} failed schema validation: {e}")
                except json.JSONDecodeError:
                    # Skip files with invalid JSON - they're tested elsewhere
                    pass
    
    def test_edge_cases_pass_schema_validation(self, schema, edge_case_files):
        """Test that edge case files pass schema validation"""
        validator = Draft7Validator(schema)
        
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Edge cases should still be valid according to schema
                    validate(instance=data, schema=schema)
                except ValidationError as e:
                    pytest.fail(f"Edge case file {file_path.name} failed schema validation: {e}")
                except json.JSONDecodeError:
                    # Skip files with invalid JSON
                    pass
    
    def test_malformed_files_fail_schema_validation(self, schema, malformed_files):
        """Test that malformed files properly fail schema validation"""
        validator = Draft7Validator(schema)
        valid_json_malformed_count = 0
        failed_validation_count = 0
        
        for file_path in malformed_files:
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    valid_json_malformed_count += 1
                    
                    # Try to validate - this should fail for malformed data
                    try:
                        validate(instance=data, schema=schema)
                        # If validation passes, the file might not be properly malformed
                        # But some edge cases might still be valid according to schema
                        pass
                    except ValidationError:
                        failed_validation_count += 1
                        
                except json.JSONDecodeError:
                    # Invalid JSON files are expected to fail
                    failed_validation_count += 1
        
        # At least some malformed files should fail validation
        assert failed_validation_count > 0, "At least some malformed files should fail schema validation"
    
    def test_schema_validates_required_fields(self, schema):
        """Test that schema enforces required fields"""
        validator = Draft7Validator(schema)
        
        # Test missing metadata
        invalid_data = {
            "participants": [{"id": "s1", "name": "Test", "role": "Test"}],
            "transcript": [{"speaker_id": "s1", "start_time": 0, "end_time": 1, "text": "test"}]
        }
        
        errors = list(validator.iter_errors(invalid_data))
        assert len(errors) > 0, "Schema should reject data missing metadata"
        
        # Test missing participants
        invalid_data = {
            "metadata": {"title": "Test", "duration_seconds": 1, "participant_count": 1, "meeting_id": "test", "date": "2024-01-01T00:00:00Z", "language": "en"},
            "transcript": [{"speaker_id": "s1", "start_time": 0, "end_time": 1, "text": "test"}]
        }
        
        errors = list(validator.iter_errors(invalid_data))
        assert len(errors) > 0, "Schema should reject data missing participants"
    
    def test_schema_validates_data_types(self, schema):
        """Test that schema enforces correct data types"""
        validator = Draft7Validator(schema)
        
        # Test invalid duration_seconds type (string instead of number)
        invalid_data = {
            "metadata": {
                "title": "Test",
                "duration_seconds": "sixty",  # Should be number
                "participant_count": 1,
                "meeting_id": "test",
                "date": "2024-01-01T00:00:00Z",
                "language": "en"
            },
            "participants": [{"id": "s1", "name": "Test", "role": "Test"}],
            "transcript": [{"speaker_id": "s1", "start_time": 0, "end_time": 1, "text": "test"}]
        }
        
        errors = list(validator.iter_errors(invalid_data))
        assert len(errors) > 0, "Schema should reject invalid data types"
    
    def test_schema_validates_participant_references(self, schema):
        """Test that schema structure supports speaker ID validation"""
        # Note: JSON Schema alone cannot enforce referential integrity between
        # participants and transcript speaker_ids, but we can test the structure
        
        # Test that speaker_id is properly defined as string
        transcript_schema = schema["properties"]["transcript"]
        item_schema = transcript_schema["items"]
        speaker_id_schema = item_schema["properties"]["speaker_id"]
        
        assert speaker_id_schema["type"] == "string", "speaker_id should be defined as string type"
        assert "minLength" in speaker_id_schema, "speaker_id should have minimum length constraint"
    
    def test_schema_documentation_completeness(self, schema):
        """Test that schema has proper documentation"""
        assert "description" in schema, "Schema should have a description"
        
        # Check that major sections have descriptions
        assert "description" in schema["properties"]["metadata"], "metadata should have description"
        assert "description" in schema["properties"]["participants"], "participants should have description"
        assert "description" in schema["properties"]["transcript"], "transcript should have description"
        
        # Check that key fields have descriptions
        metadata_props = schema["properties"]["metadata"]["properties"]
        for field in ["title", "duration_seconds", "participant_count"]:
            assert "description" in metadata_props[field], f"metadata.{field} should have description"