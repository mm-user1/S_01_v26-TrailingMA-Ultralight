"""
Sanity tests to verify test infrastructure is working correctly.
These tests ensure basic imports and environment setup function properly.
"""

import pytest
import sys
import os
from pathlib import Path


class TestSanityChecks:
    """Basic sanity checks for test infrastructure."""

    def test_python_version(self):
        """Ensure Python version is 3.8+."""
        assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version_info}"

    def test_pytest_working(self):
        """Verify pytest is functioning."""
        assert True

    def test_imports_work(self):
        """Test that basic project imports work."""
        try:
            import core.backtest_engine
            import core.optuna_engine
            import core.walkforward_engine
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")

    def test_data_directory_exists(self):
        """Verify data directory structure exists."""
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        assert data_dir.exists(), "data/ directory not found"

        # Check for raw data directory
        raw_dir = data_dir / "raw"
        assert raw_dir.exists(), "data/raw/ directory not found"

    def test_baseline_directory_exists(self):
        """Verify baseline directory for regression tests exists."""
        project_root = Path(__file__).parent.parent
        baseline_dir = project_root / "data" / "baseline"
        assert baseline_dir.exists(), "data/baseline/ directory not found"

    def test_src_directory_structure(self):
        """Verify src directory contains expected files."""
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"

        expected_files = [
            Path("core") / "backtest_engine.py",
            Path("core") / "optuna_engine.py",
            Path("core") / "walkforward_engine.py",
            Path("ui") / "server.py",
            Path("ui") / "templates" / "index.html",
            Path("ui") / "static" / "css" / "style.css",
            Path("ui") / "static" / "js" / "utils.js",
            Path("ui") / "static" / "js" / "api.js",
            Path("ui") / "static" / "js" / "strategy-config.js",
            Path("ui") / "static" / "js" / "presets.js",
            Path("ui") / "static" / "js" / "ui-handlers.js",
            Path("ui") / "static" / "js" / "main.js",
        ]

        for file_name in expected_files:
            file_path = src_dir / file_name
            assert file_path.exists(), f"Expected file {file_name} not found in src/"


class TestDependencies:
    """Test that required dependencies are installed."""

    def test_pandas_available(self):
        """Test pandas is installed."""
        try:
            import pandas as pd
            assert hasattr(pd, 'DataFrame')
        except ImportError:
            pytest.fail("pandas not installed")

    def test_numpy_available(self):
        """Test numpy is installed."""
        try:
            import numpy as np
            assert hasattr(np, 'array')
        except ImportError:
            pytest.fail("numpy not installed")

    def test_optuna_available(self):
        """Test optuna is installed."""
        try:
            import optuna
            # Just verify it's installed and has expected attributes
            assert hasattr(optuna, 'create_study')
        except ImportError:
            pytest.fail("optuna not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
