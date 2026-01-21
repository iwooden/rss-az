"""pytest configuration for tests."""
import sys
from pathlib import Path

# Add project root to Python path so we can import core, entities, etc.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
