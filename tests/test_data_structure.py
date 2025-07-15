"""
Test data structure validation for Task 2.1
Validates that all required test data directories exist
"""
import os
import pytest
from pathlib import Path


class TestDataStructure:
    """Test that test data directory structure is properly organized"""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory"""
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def test_data_root(self, project_root):
        """Get test data root directory"""
        return project_root / "test_data"
    
    def test_test_data_directory_exists(self, test_data_root):
        """Test that test_data/ directory exists"""
        assert test_data_root.exists(), "test_data/ directory must exist"
        assert test_data_root.is_dir(), "test_data/ must be a directory"
    
    def test_samples_directory_exists(self, test_data_root):
        """Test that test_data/samples/ directory exists"""
        samples_dir = test_data_root / "samples"
        assert samples_dir.exists(), "test_data/samples/ directory must exist"
        assert samples_dir.is_dir(), "test_data/samples/ must be a directory"
    
    def test_edge_cases_directory_exists(self, test_data_root):
        """Test that test_data/edge_cases/ directory exists"""
        edge_cases_dir = test_data_root / "edge_cases"
        assert edge_cases_dir.exists(), "test_data/edge_cases/ directory must exist"
        assert edge_cases_dir.is_dir(), "test_data/edge_cases/ must be a directory"
    
    def test_malformed_directory_exists(self, test_data_root):
        """Test that test_data/malformed/ directory exists"""
        malformed_dir = test_data_root / "malformed"
        assert malformed_dir.exists(), "test_data/malformed/ directory must exist"
        assert malformed_dir.is_dir(), "test_data/malformed/ must be a directory"
    
    def test_performance_directory_exists(self, test_data_root):
        """Test that test_data/performance/ directory exists"""
        performance_dir = test_data_root / "performance"
        assert performance_dir.exists(), "test_data/performance/ directory must exist"
        assert performance_dir.is_dir(), "test_data/performance/ must be a directory"
    
    def test_schemas_directory_exists(self, test_data_root):
        """Test that test_data/schemas/ directory exists"""
        schemas_dir = test_data_root / "schemas"
        assert schemas_dir.exists(), "test_data/schemas/ directory must exist"
        assert schemas_dir.is_dir(), "test_data/schemas/ must be a directory"
    
    def test_all_required_directories_exist(self, test_data_root):
        """Test that all required subdirectories exist"""
        required_dirs = ["samples", "edge_cases", "malformed", "performance", "schemas"]
        
        for dir_name in required_dirs:
            dir_path = test_data_root / dir_name
            assert dir_path.exists(), f"test_data/{dir_name}/ directory must exist"
            assert dir_path.is_dir(), f"test_data/{dir_name}/ must be a directory"
    
    def test_directory_structure_completeness(self, test_data_root):
        """Test that the directory structure is complete and organized"""
        # Get all subdirectories
        subdirs = [d.name for d in test_data_root.iterdir() if d.is_dir()]
        
        # Required directories
        required = {"samples", "edge_cases", "malformed", "performance", "schemas"}
        
        # Check that all required directories exist
        missing = required - set(subdirs)
        assert not missing, f"Missing required directories: {missing}"
        
        # Verify we have at least the required directories
        assert len(subdirs) >= len(required), "Not enough directories in test_data/"