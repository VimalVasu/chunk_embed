"""
Test sample data validation for Task 2.2
Validates JSON structure and content of sample transcript files
"""
import json
import pytest
from pathlib import Path


class TestSampleData:
    """Test that sample transcript data has valid JSON structure and content"""
    
    @pytest.fixture
    def samples_dir(self):
        """Get samples directory"""
        return Path(__file__).parent.parent / "test_data" / "samples"
    
    @pytest.fixture
    def sample_files(self, samples_dir):
        """Get all JSON sample files"""
        return list(samples_dir.glob("*.json"))
    
    def test_samples_directory_has_files(self, sample_files):
        """Test that samples directory contains JSON files"""
        assert len(sample_files) > 0, "samples/ directory must contain at least one JSON file"
    
    def test_all_sample_files_are_valid_json(self, sample_files):
        """Test that all sample files contain valid JSON"""
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {file_path.name}: {e}")
    
    def test_sample_files_have_required_structure(self, sample_files):
        """Test that sample files have required top-level structure"""
        required_keys = {"metadata", "participants", "transcript"}
        
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"{file_path.name} missing required keys: {missing_keys}"
    
    def test_metadata_structure(self, sample_files):
        """Test that metadata has required fields"""
        required_metadata_keys = {"title", "duration_seconds", "participant_count", "meeting_id", "date", "language"}
        
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metadata = data["metadata"]
            missing_keys = required_metadata_keys - set(metadata.keys())
            assert not missing_keys, f"{file_path.name} metadata missing keys: {missing_keys}"
            
            # Validate data types
            assert isinstance(metadata["duration_seconds"], (int, float)), f"{file_path.name}: duration_seconds must be numeric"
            assert isinstance(metadata["participant_count"], int), f"{file_path.name}: participant_count must be integer"
            assert isinstance(metadata["title"], str), f"{file_path.name}: title must be string"
            assert isinstance(metadata["meeting_id"], str), f"{file_path.name}: meeting_id must be string"
            assert isinstance(metadata["date"], str), f"{file_path.name}: date must be string"
            assert isinstance(metadata["language"], str), f"{file_path.name}: language must be string"
    
    def test_participants_structure(self, sample_files):
        """Test that participants have required fields"""
        required_participant_keys = {"id", "name", "role"}
        
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            participants = data["participants"]
            assert isinstance(participants, list), f"{file_path.name}: participants must be a list"
            assert len(participants) > 0, f"{file_path.name}: must have at least one participant"
            
            for i, participant in enumerate(participants):
                missing_keys = required_participant_keys - set(participant.keys())
                assert not missing_keys, f"{file_path.name} participant {i} missing keys: {missing_keys}"
                
                # Validate data types
                assert isinstance(participant["id"], str), f"{file_path.name}: participant id must be string"
                assert isinstance(participant["name"], str), f"{file_path.name}: participant name must be string"
                assert isinstance(participant["role"], str), f"{file_path.name}: participant role must be string"
    
    def test_transcript_structure(self, sample_files):
        """Test that transcript entries have required fields"""
        required_transcript_keys = {"speaker_id", "start_time", "end_time", "text"}
        
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            transcript = data["transcript"]
            assert isinstance(transcript, list), f"{file_path.name}: transcript must be a list"
            assert len(transcript) > 0, f"{file_path.name}: transcript must have at least one entry"
            
            for i, entry in enumerate(transcript):
                missing_keys = required_transcript_keys - set(entry.keys())
                assert not missing_keys, f"{file_path.name} transcript entry {i} missing keys: {missing_keys}"
                
                # Validate data types
                assert isinstance(entry["speaker_id"], str), f"{file_path.name}: speaker_id must be string"
                assert isinstance(entry["start_time"], (int, float)), f"{file_path.name}: start_time must be numeric"
                assert isinstance(entry["end_time"], (int, float)), f"{file_path.name}: end_time must be numeric"
                assert isinstance(entry["text"], str), f"{file_path.name}: text must be string"
                
                # Validate timing logic
                assert entry["start_time"] >= 0, f"{file_path.name}: start_time must be non-negative"
                assert entry["end_time"] > entry["start_time"], f"{file_path.name}: end_time must be after start_time"
    
    def test_speaker_id_consistency(self, sample_files):
        """Test that speaker_ids in transcript match participant ids"""
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            participant_ids = {p["id"] for p in data["participants"]}
            transcript_speaker_ids = {entry["speaker_id"] for entry in data["transcript"]}
            
            # All transcript speaker_ids must be in participants
            invalid_speakers = transcript_speaker_ids - participant_ids
            assert not invalid_speakers, f"{file_path.name}: unknown speaker_ids in transcript: {invalid_speakers}"
    
    def test_timestamp_ordering(self, sample_files):
        """Test that transcript entries are in chronological order"""
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            transcript = data["transcript"]
            for i in range(1, len(transcript)):
                prev_start = transcript[i-1]["start_time"]
                curr_start = transcript[i]["start_time"]
                assert curr_start >= prev_start, f"{file_path.name}: transcript not in chronological order at entry {i}"
    
    def test_duration_consistency(self, sample_files):
        """Test that metadata duration matches transcript timing"""
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metadata_duration = data["metadata"]["duration_seconds"]
            transcript = data["transcript"]
            
            if transcript:
                max_end_time = max(entry["end_time"] for entry in transcript)
                # Allow some tolerance for rounding
                assert abs(metadata_duration - max_end_time) <= 5, \
                    f"{file_path.name}: metadata duration ({metadata_duration}) doesn't match transcript max time ({max_end_time})"
    
    def test_participant_count_consistency(self, sample_files):
        """Test that metadata participant_count matches actual participants"""
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metadata_count = data["metadata"]["participant_count"]
            actual_count = len(data["participants"])
            
            assert metadata_count == actual_count, \
                f"{file_path.name}: metadata participant_count ({metadata_count}) doesn't match actual count ({actual_count})"
    
    def test_realistic_content_quality(self, sample_files):
        """Test that sample files contain realistic, varied content"""
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check for realistic duration (1-5 minutes as specified)
            duration = data["metadata"]["duration_seconds"]
            assert 60 <= duration <= 300, f"{file_path.name}: duration should be 1-5 minutes (60-300 seconds)"
            
            # Check for meaningful text content
            transcript = data["transcript"]
            total_text_length = sum(len(entry["text"]) for entry in transcript)
            assert total_text_length > 100, f"{file_path.name}: transcript text seems too short for realistic content"
            
            # Check for varied speakers if multi-participant
            if data["metadata"]["participant_count"] > 1:
                unique_speakers = {entry["speaker_id"] for entry in transcript}
                assert len(unique_speakers) > 1, f"{file_path.name}: multi-participant meeting should have multiple speakers in transcript"
    
    def test_variety_in_sample_data(self, sample_files):
        """Test that sample files demonstrate variety in speakers, topics, and lengths (Task 2.3)"""
        assert len(sample_files) >= 3, "Must have at least 3 sample files to demonstrate variety"
        
        # Collect metadata from all files
        all_data = []
        for file_path in sample_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.append((file_path.name, data))
        
        # Test variety in participant counts (different speakers)
        participant_counts = [data["metadata"]["participant_count"] for _, data in all_data]
        unique_participant_counts = set(participant_counts)
        assert len(unique_participant_counts) >= 2, f"Must have variety in participant counts. Found: {unique_participant_counts}"
        
        # Test variety in durations (different lengths)
        durations = [data["metadata"]["duration_seconds"] for _, data in all_data]
        min_duration = min(durations)
        max_duration = max(durations)
        # Should have at least 60 seconds difference to show variety in lengths
        assert max_duration - min_duration >= 60, f"Must have variety in durations. Range: {min_duration}-{max_duration}"
        
        # Test variety in topics (different meeting types based on titles)
        titles = [data["metadata"]["title"].lower() for _, data in all_data]
        
        # Look for different meeting types in titles
        meeting_types = set()
        for title in titles:
            if any(keyword in title for keyword in ["standup", "daily"]):
                meeting_types.add("standup")
            elif any(keyword in title for keyword in ["interview", "technical"]):
                meeting_types.add("interview")
            elif any(keyword in title for keyword in ["presentation", "sales", "results"]):
                meeting_types.add("presentation")
            elif any(keyword in title for keyword in ["training", "product"]):
                meeting_types.add("training")
            elif any(keyword in title for keyword in ["support", "customer", "call"]):
                meeting_types.add("support")
            else:
                meeting_types.add("other")
        
        assert len(meeting_types) >= 3, f"Must have variety in meeting types. Found types: {meeting_types}"
        
        # Test variety in roles across all participants
        all_roles = set()
        for _, data in all_data:
            for participant in data["participants"]:
                all_roles.add(participant["role"].lower())
        
        assert len(all_roles) >= 5, f"Must have variety in participant roles. Found roles: {all_roles}"