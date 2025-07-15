"""
Test edge case data validation for Task 2.4
Validates edge cases: overlapping speech, long monologues, silence gaps
"""
import json
import pytest
from pathlib import Path


class TestEdgeCases:
    """Test that edge case transcript data handles challenging scenarios"""
    
    @pytest.fixture
    def edge_cases_dir(self):
        """Get edge_cases directory"""
        return Path(__file__).parent.parent / "test_data" / "edge_cases"
    
    @pytest.fixture
    def edge_case_files(self, edge_cases_dir):
        """Get all JSON edge case files"""
        return list(edge_cases_dir.glob("*.json"))
    
    def test_edge_cases_directory_has_files(self, edge_case_files):
        """Test that edge_cases directory contains JSON files"""
        assert len(edge_case_files) > 0, "edge_cases/ directory must contain at least one JSON file"
    
    def test_all_edge_case_files_are_valid_json(self, edge_case_files):
        """Test that all edge case files contain valid JSON"""
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {file_path.name}: {e}")
    
    def test_edge_case_files_have_metadata_type(self, edge_case_files):
        """Test that edge case files specify their edge case type"""
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            assert "edge_case_type" in data["metadata"], f"{file_path.name} must specify edge_case_type in metadata"
            
            edge_case_type = data["metadata"]["edge_case_type"]
            valid_types = {"overlapping_speech", "long_monologue", "silence_gaps"}
            assert edge_case_type in valid_types, f"{file_path.name}: edge_case_type must be one of {valid_types}"
    
    def test_overlapping_speech_cases(self, edge_case_files):
        """Test overlapping speech edge cases have proper overlapping timestamps"""
        overlapping_files = []
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if data["metadata"].get("edge_case_type") == "overlapping_speech":
                overlapping_files.append((file_path, data))
        
        assert len(overlapping_files) > 0, "Must have at least one overlapping speech edge case"
        
        for file_path, data in overlapping_files:
            transcript = data["transcript"]
            overlaps_found = 0
            
            # Check for overlapping speech (where end_time of one entry > start_time of next)
            for i in range(len(transcript) - 1):
                current_end = transcript[i]["end_time"]
                next_start = transcript[i + 1]["start_time"]
                
                if current_end > next_start:
                    overlaps_found += 1
            
            assert overlaps_found > 0, f"{file_path.name}: overlapping speech case must have actual overlapping timestamps"
    
    def test_long_monologue_cases(self, edge_case_files):
        """Test long monologue edge cases have extended single speaker segments"""
        monologue_files = []
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if data["metadata"].get("edge_case_type") == "long_monologue":
                monologue_files.append((file_path, data))
        
        assert len(monologue_files) > 0, "Must have at least one long monologue edge case"
        
        for file_path, data in monologue_files:
            transcript = data["transcript"]
            
            # Find the longest single transcript entry
            max_duration = 0
            max_text_length = 0
            
            for entry in transcript:
                duration = entry["end_time"] - entry["start_time"]
                text_length = len(entry["text"])
                
                max_duration = max(max_duration, duration)
                max_text_length = max(max_text_length, text_length)
            
            # Long monologue should have at least one segment > 60 seconds
            assert max_duration > 60, f"{file_path.name}: long monologue must have at least one segment > 60 seconds"
            
            # Long monologue should have substantial text content
            assert max_text_length > 500, f"{file_path.name}: long monologue must have substantial text content"
    
    def test_silence_gaps_cases(self, edge_case_files):
        """Test silence gaps edge cases have significant gaps between speech"""
        silence_files = []
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if data["metadata"].get("edge_case_type") == "silence_gaps":
                silence_files.append((file_path, data))
        
        assert len(silence_files) > 0, "Must have at least one silence gaps edge case"
        
        for file_path, data in silence_files:
            transcript = data["transcript"]
            significant_gaps_found = 0
            
            # Check for significant gaps between transcript entries
            for i in range(len(transcript) - 1):
                current_end = transcript[i]["end_time"]
                next_start = transcript[i + 1]["start_time"]
                gap = next_start - current_end
                
                # Consider a gap > 10 seconds as significant
                if gap > 10:
                    significant_gaps_found += 1
            
            assert significant_gaps_found > 0, f"{file_path.name}: silence gaps case must have significant gaps (>10s) between speech"
    
    def test_edge_cases_basic_structure(self, edge_case_files):
        """Test that edge cases still maintain basic transcript structure"""
        required_keys = {"metadata", "participants", "transcript"}
        
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"{file_path.name} missing required keys: {missing_keys}"
            
            # Basic validation that transcript has required fields
            transcript = data["transcript"]
            assert isinstance(transcript, list), f"{file_path.name}: transcript must be a list"
            assert len(transcript) > 0, f"{file_path.name}: transcript must have at least one entry"
            
            required_transcript_keys = {"speaker_id", "start_time", "end_time", "text"}
            for i, entry in enumerate(transcript):
                missing_entry_keys = required_transcript_keys - set(entry.keys())
                assert not missing_entry_keys, f"{file_path.name} transcript entry {i} missing keys: {missing_entry_keys}"
    
    def test_edge_cases_have_variety(self, edge_case_files):
        """Test that we have all three types of edge cases represented"""
        edge_case_types = set()
        
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "edge_case_type" in data["metadata"]:
                edge_case_types.add(data["metadata"]["edge_case_type"])
        
        required_types = {"overlapping_speech", "long_monologue", "silence_gaps"}
        missing_types = required_types - edge_case_types
        assert not missing_types, f"Missing edge case types: {missing_types}"
    
    def test_edge_cases_realistic_content(self, edge_case_files):
        """Test that edge cases contain realistic content despite edge characteristics"""
        for file_path in edge_case_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check for non-empty text content
            transcript = data["transcript"]
            for entry in transcript:
                assert entry["text"].strip(), f"{file_path.name}: transcript entries must have non-empty text"
                assert len(entry["text"]) > 5, f"{file_path.name}: transcript text should be meaningful (>5 chars)"
            
            # Check for realistic timing
            for entry in transcript:
                assert entry["start_time"] >= 0, f"{file_path.name}: start_time must be non-negative"
                assert entry["end_time"] > entry["start_time"], f"{file_path.name}: end_time must be after start_time"