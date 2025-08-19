"""
Basic tests for the main application module.
"""
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_main_file_exists():
    """Test that main.py exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(project_root, "main.py")
    assert os.path.exists(main_path)

def test_main_file_readable():
    """Test that main.py is readable."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(project_root, "main.py")
    
    with open(main_path, 'r') as f:
        content = f.read()
        assert len(content) > 0
        assert 'streamlit' in content

def test_app_modules_directory():
    """Test that app_modules directory exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_modules_path = os.path.join(project_root, "app_modules")
    assert os.path.exists(app_modules_path)
    assert os.path.isdir(app_modules_path)

def test_config_file_exists():
    """Test that config.py exists in app_modules."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "app_modules", "config.py")
    assert os.path.exists(config_path)
