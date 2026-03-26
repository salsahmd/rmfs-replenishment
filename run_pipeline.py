"""
RMFS warehouse simulation pipeline (Rika's baseline).

Usage:
    python run_pipeline.py [--max-ticks 1000]

This delegates to pipeline/run_baseline.py.
"""

import os
import sys

# Add pipeline/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))

from run_baseline import main

if __name__ == "__main__":
    main()
