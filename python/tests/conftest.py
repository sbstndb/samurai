import sys
import os

# Add build directory to Python path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build_py314', 'python'))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)
