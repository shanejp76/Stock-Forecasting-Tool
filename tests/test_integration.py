"""
Integration tests for the Stock Forecasting Tool.
"""
import os
import sys
from unittest.mock import Mock, patch

# Add the app_modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestApplicationIntegration:
    """Test application integration points."""
    
    def test_import_main_modules(self):
        """Test that main application modules can be imported."""
        try:
            import main
            assert hasattr(main, 'ALPHA_VANTAGE_API_KEY')
        except ImportError as e:
            # If streamlit or other dependencies are missing, skip
            assert 'streamlit' in str(e) or 'alpha_vantage' in str(e)
    
    def test_app_modules_structure(self):
        """Test that app_modules package has expected structure."""
        expected_modules = [
            'config',
            'data_handler',
            'data_pipeline',
            'forecast_summary',
            'model_orchestrator',
            'performance_metrics',
            'plotter',
            'ui_intro'
        ]
        
        app_modules_path = os.path.join(os.path.dirname(__file__), '..', 'app_modules')
        
        for module in expected_modules:
            module_file = f"{module}.py"
            assert os.path.exists(os.path.join(app_modules_path, module_file)), f"Missing {module_file}"
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and has content."""
        req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        assert os.path.exists(req_path)
        
        with open(req_path, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert 'streamlit' in content
            assert 'pandas' in content
    
    def test_docker_files_exist(self):
        """Test that Docker-related files exist."""
        base_path = os.path.join(os.path.dirname(__file__), '..')
        
        docker_files = [
            'Dockerfile',
            'Dockerfile.prod',
            'docker-compose.yml',
            '.dockerignore'
        ]
        
        for docker_file in docker_files:
            assert os.path.exists(os.path.join(base_path, docker_file)), f"Missing {docker_file}"


class TestEnvironmentSetup:
    """Test environment setup and configuration."""
    
    def test_env_example_file(self):
        """Test that .env.example file exists and has required variables."""
        env_example_path = os.path.join(os.path.dirname(__file__), '..', '.env.example')
        assert os.path.exists(env_example_path)
        
        with open(env_example_path, 'r') as f:
            content = f.read()
            assert 'ALPHA_VANTAGE_API_KEY' in content
            assert 'FINNHUB_API_KEY' in content
    
    def test_scripts_directory(self):
        """Test that deployment scripts exist."""
        scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        assert os.path.exists(scripts_path)
        
        expected_scripts = [
            'deploy-local.sh',
            'deploy-cloud.sh'
        ]
        
        for script in expected_scripts:
            script_path = os.path.join(scripts_path, script)
            assert os.path.exists(script_path), f"Missing {script}"
            
            # Check if script is executable (on Unix systems)
            if os.name != 'nt':  # Not Windows
                assert os.access(script_path, os.X_OK), f"{script} is not executable"


class TestDocumentation:
    """Test documentation files."""
    
    def test_documentation_files_exist(self):
        """Test that documentation files exist."""
        base_path = os.path.join(os.path.dirname(__file__), '..')
        
        doc_files = [
            'README.md',
            'MODERNIZATION_ROADMAP.md',
            'DOCKER_DEPLOYMENT.md'
        ]
        
        for doc_file in doc_files:
            doc_path = os.path.join(base_path, doc_file)
            assert os.path.exists(doc_path), f"Missing {doc_file}"
            
            # Check that files have content
            with open(doc_path, 'r') as f:
                content = f.read()
                assert len(content) > 100, f"{doc_file} seems too short"
    
    def test_github_workflows(self):
        """Test that GitHub workflows exist."""
        workflow_path = os.path.join(os.path.dirname(__file__), '..', '.github', 'workflows')
        
        if os.path.exists(workflow_path):
            workflow_files = os.listdir(workflow_path)
            assert len(workflow_files) > 0, "No workflow files found"
            
            # Check for CI/CD workflow
            ci_cd_exists = any('ci' in f.lower() for f in workflow_files)
            assert ci_cd_exists, "No CI/CD workflow found"
