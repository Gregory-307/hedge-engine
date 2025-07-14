import sys
from pathlib import Path

# Ensure project root on sys.path for local 'pytest' execution without installation
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root)) 