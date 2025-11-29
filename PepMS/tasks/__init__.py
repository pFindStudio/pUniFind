import importlib
from pathlib import Path

# automatically import any Python files in the criterions/ directory
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("PepMS.tasks." + file.name[:-3])
