from __future__ import annotations

import sys
from pathlib import Path


CSG_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CSG_ROOT.parent
DATA_ROOT = REPOSITORY_ROOT / "data"

if str(CSG_ROOT) not in sys.path:
    sys.path.insert(0, str(CSG_ROOT))
