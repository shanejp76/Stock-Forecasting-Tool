"""
Environment Check Script for Stock Forecasting Tool

This script verifies that all required dependencies are properly installed
and the environment is correctly configured.

Usage: python check_env.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 9:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.9+ required")
        return False


def check_conda_environment():
    """Check if running in conda environment"""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        print(f"✅ Running in conda environment: {conda_env}")
        return True
    elif "conda" in sys.executable:
        print("✅ Running in conda environment (detected from path)")
        return True
    else:
        print("❌ Not running in conda environment")
        return False


def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = [
        ("streamlit", "Streamlit web framework"),
        ("google.cloud.bigquery", "Google BigQuery client"),
        ("pandas", "Data manipulation library"),
        ("numpy", "Numerical computing library"),
        ("python_dotenv", "Environment variable loading"),
    ]

    all_available = True

    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (NOT INSTALLED)")
            all_available = False

    return all_available


def check_custom_modules():
    """Check if custom application modules are available"""
    custom_modules = [
        "app_modules.bigquery_client",
        "app_modules.data_handler",
        "app_modules.config",
    ]

    all_available = True

    for module in custom_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - Error: {str(e)}")
            all_available = False

    return all_available


def check_environment_files():
    """Check if required configuration files exist"""
    required_files = [
        (".env", "Environment variables file"),
        ("requirements.txt", "Python dependencies"),
        ("main.py", "Streamlit application entry point"),
    ]

    all_present = True

    for filename, description in required_files:
        if Path(filename).exists():
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} (MISSING)")
            all_present = False

    return all_present


def check_bigquery_connection():
    """Test BigQuery connection"""
    try:
        from app_modules.bigquery_client import BigQueryClient

        client = BigQueryClient()
        symbols = client.get_available_symbols()
        print(f"✅ BigQuery connection successful - {len(symbols)} symbols available")
        return True
    except Exception as e:
        print(f"❌ BigQuery connection failed: {str(e)}")
        return False


def main():
    """Run all environment checks"""
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 50)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Conda Environment", check_conda_environment),
        ("Required Packages", check_required_packages),
        ("Custom Modules", check_custom_modules),
        ("Configuration Files", check_environment_files),
        ("BigQuery Connection", check_bigquery_connection),
    ]

    results = []

    for check_name, check_function in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name))
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {str(e)}")
            results.append((check_name, False))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 Environment is ready for development!")
        print("\nNext steps:")
        print("  python -m streamlit run main.py  # Launch application")
    else:
        print(f"\n⚠️ {total - passed} issues need to be resolved")
        print("\nRecommended fixes:")
        print("  pip install -r requirements.txt  # Install missing packages")
        print("  conda activate stock-forecasting  # Activate environment")


if __name__ == "__main__":
    main()
