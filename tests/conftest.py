import sys
from pathlib import Path

# tests import the inner package (h3_tools/) straight from the repo root,
# with no ComfyUI on the path — refs.py must stay stdlib-pure.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
