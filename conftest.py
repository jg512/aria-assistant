"""
conftest.py — shared pytest configuration
Suppresses hardware-specific import errors so tests run without a Pi.
"""
import sys
from unittest.mock import MagicMock

# Mock hardware-dependent libraries so tests run on any machine
MOCK_MODULES = [
    "pyaudio",
    "faster_whisper",
    "openwakeword",
    "openwakeword.model",
    "numpy",
    "psutil",
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# numpy needs to behave well in a few places
import numpy as np_mock
sys.modules["numpy"] = np_mock if hasattr(np_mock, "frombuffer") else MagicMock()
