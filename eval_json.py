import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_module("self_reflection_llm.eval_json", run_name="__main__")
