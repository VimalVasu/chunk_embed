"""
Test performance data validation for Task 2.8
Validates performance test datasets (larger files)
"""
import json
import pytest
from pathlib import Path


class TestPerformanceData:
    """Test that performance datasets are properly structured for testing"""
    
    @pytest.fixture
    def performance_dir(self):
        """Get performance directory"""
        return Path(__file__).parent.parent / "test_data" / "performance"
    
    @pytest.fixture
    def performance_files(self, performance_dir):
        """Get all performance test files"""
        return list(performance_dir.glob("*.json"))
    
    def test_performance_directory_has_files(self, performance_files):
        """Test that performance directory contains test files"""
        assert len(performance_files) > 0, "performance/ directory must contain at least one test file"
    
    def test_performance_files_are_larger(self, performance_files):
        """Test that performance files are larger than regular samples"""
        for file_path in performance_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Performance files should have longer duration
            duration = data["metadata"]["duration_seconds"]
            assert duration >= 300, f"Performance file {file_path.name} should be at least 5 minutes (300s)"
            
            # Should have more transcript entries
            transcript = data["transcript"]
            assert len(transcript) >= 5, f"Performance file {file_path.name} should have substantial transcript entries"
    
    def test_performance_files_valid_structure(self, performance_files):
        """Test that performance files maintain valid structure"""
        required_keys = {"metadata", "participants", "transcript"}
        
        for file_path in performance_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"Performance file {file_path.name} missing keys: {missing_keys}"