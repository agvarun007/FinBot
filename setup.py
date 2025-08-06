"""
FinBot Setup Validation Script

Validates the development environment and configuration before running FinBot.
Checks dependencies, configuration, and database connectivity.
"""

import os
import sys
from pathlib import Path


def check_python_version() -> bool:
    """Verify Python version meets minimum requirements."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_environment_file() -> bool:
    """Verify .env configuration file exists."""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("💡 Copy .env.example to .env and configure your settings")
        return False
    print("✅ .env file found")
    return True


def check_required_packages() -> bool:
    """Verify all required Python packages are installed."""
    # Map pip package names to their import names
    required_packages = {
        "psycopg2": "psycopg2",
        "pgvector": "pgvector", 
        "sentence_transformers": "sentence_transformers",
        "transformers": "transformers",
        "rich": "rich",
        "pdfplumber": "pdfplumber",
        "beautifulsoup4": "bs4",  # beautifulsoup4 imports as bs4
        "python-dotenv": "dotenv",  # python-dotenv imports as dotenv
        "llama-cpp-python": "llama_cpp",  # llama-cpp-python imports as llama_cpp
        "openai": "openai"
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} (import: {import_name})")
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True


def check_llm_configuration() -> bool:
    """Verify LLM backend configuration is valid."""
    # Skip validation during config check to avoid circular imports
    os.environ["SKIP_CONFIG_VALIDATION"] = "1"
    
    try:
        from finbot.config import LLM_BACKEND, LLAMA_PATH
        
        if LLM_BACKEND == "llama_cpp":
            if not LLAMA_PATH:
                print("❌ LLAMA_PATH not set in .env")
                return False
            if not Path(LLAMA_PATH).exists():
                print(f"❌ Model file not found: {LLAMA_PATH}")
                return False
            print(f"✅ Llama model found: {Path(LLAMA_PATH).name}")
        
        elif LLM_BACKEND == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ OPENAI_API_KEY not set in .env")
                return False
            print("✅ OpenAI API key configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def initialize_database() -> bool:
    """Initialize the database if not already set up."""
    try:
        print("🔄 Initializing database...")
        from scripts.init_db import initialize_database
        initialize_database()
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def main():
    """Run comprehensive setup validation."""
    print("🤖 FinBot Setup Validation")
    print("=" * 50)
    
    validation_checks = [
        ("Python Version", check_python_version),
        ("Environment Configuration", check_environment_file),
        ("Python Dependencies", check_required_packages),
        ("LLM Configuration", check_llm_configuration),
        ("Database Setup", initialize_database),
    ]
    
    all_checks_passed = True
    
    for check_name, check_function in validation_checks:
        print(f"\n📋 {check_name}:")
        if not check_function():
            all_checks_passed = False
    
    print("\n" + "=" * 50)
    
    if all_checks_passed:
        print("🎉 Setup validation completed successfully!")
        print("\n📚 Next steps:")
        print("1. Add documents to data/raw/ directory")
        print("2. Run: python -m finbot.cli --ingest")
        print("3. Start interactive mode: python -m finbot.cli")
    else:
        print("❌ Setup validation failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
