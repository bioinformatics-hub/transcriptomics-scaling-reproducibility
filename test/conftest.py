"""Keep the public project root importable during local regression tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ["PATH_TO_TEST_CONFIGS"] = str(Path(__file__).resolve().parent / "test_configs")
