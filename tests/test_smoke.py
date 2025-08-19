"""
Basic smoke tests to ensure the CI/CD pipeline works.
"""
import sys
import os

def test_python_version():
    """Test that we're running the expected Python version."""
    assert sys.version_info >= (3, 11)

def test_imports_basic():
    """Test that basic Python imports work."""
    import json
    import datetime
    import pathlib
    assert True

def test_project_structure():
    """Test that the project has the expected structure."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for key files
    assert os.path.exists(os.path.join(project_root, "main.py"))
    assert os.path.exists(os.path.join(project_root, "requirements.txt"))
    assert os.path.exists(os.path.join(project_root, "Dockerfile"))
    assert os.path.exists(os.path.join(project_root, "app_modules"))

def test_environment_variables():
    """Test that environment variables can be read."""
    # Just test that os.environ works
    import os
    env_vars = os.environ
    assert isinstance(env_vars, dict)

def test_requirements_file():
    """Test that requirements.txt exists and is readable."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_path = os.path.join(project_root, "requirements.txt")
    
    assert os.path.exists(requirements_path)
    
    with open(requirements_path, 'r') as f:
        content = f.read()
        assert len(content) > 0
        assert 'streamlit' in content.lower()
