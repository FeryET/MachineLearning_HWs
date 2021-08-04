import os
import sys
from pathlib import Path

fpath = Path(os.path.realpath(__file__))
sys.path.append(fpath.parent.stem)