import sys
import os

# Add build directory to Python path for pytest
# Try multiple possible build locations
possible_build_dirs = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build_py314', 'python')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'python')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build_py314', 'python')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')),
]

for build_python_dir in possible_build_dirs:
    if os.path.exists(build_python_dir) and build_python_dir not in sys.path:
        sys.path.insert(0, build_python_dir)
        break
