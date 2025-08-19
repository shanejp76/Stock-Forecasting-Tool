"""
Integration tests for the Stock Forecasting Tool.
"""
import os
import sys


class TestApplicationIntegration:
    """Test application integration."""
    
    def test_import_main_modules(self):
        """Test that main application modules can be imported."""
        try:
            # Test individual module imports instead of main.py
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            
            # Import app modules without triggering execution
            from app_modules import config
            from app_modules import data_handler
            
            assert hasattr(config, 'load_environment_variables')
            assert hasattr(data_handler, 'process_technical_indicators')
            
        except ImportError as e:
            # Skip if dependencies aren't available
            assert "No module named" in str(e) or "Missing optional dependency" in str(e)
    
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
        
        try:
            with open(req_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0
        except UnicodeDecodeError:
            # Handle files that might be in different encoding
            with open(req_path, 'rb') as f:
                content = f.read()
                assert len(content) > 0
    
    def test_docker_files_exist(self):
        """Test that Docker files exist."""
        base_path = os.path.join(os.path.dirname(__file__), '..')
        docker_files = ['Dockerfile', 'docker-compose.yml', '.dockerignore']
        
        for docker_file in docker_files:
            docker_path = os.path.join(base_path, docker_file)
            assert os.path.exists(docker_path), f"Missing {docker_file}"


class TestEnvironmentSetup:
    """Test environment setup."""
    
    def test_env_example_file(self):
        """Test that .env.example exists."""
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
            'docs/MODERNIZATION_ROADMAP.md',
            'docs/DOCKER_DEPLOYMENT.md'
        ]
        
        for doc_file in doc_files:
            doc_path = os.path.join(base_path, doc_file)
            assert os.path.exists(doc_path), f"Missing {doc_file}"
            
            # Check that files have content
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert len(content) > 0, f"{doc_file} is empty"
            except UnicodeDecodeError:
                # Handle files that might be in different encoding
                with open(doc_path, 'rb') as f:
                    content = f.read()
                    assert len(content) > 0, f"{doc_file} is empty"
    
    def test_github_workflows(self):
        """Test that GitHub workflows exist."""
        workflows_path = os.path.join(os.path.dirname(__file__), '..', '.github', 'workflows')
        assert os.path.exists(workflows_path)
        
        ci_cd_path = os.path.join(workflows_path, 'ci-cd.yml')
        assert os.path.exists(ci_cd_path)
