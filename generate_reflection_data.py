import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility dispatcher for the split reflection-data generation scripts."
    )
    parser.add_argument(
        "--format",
        choices=["paragraph", "code", "table"],
        required=True,
        help="Select which split pipeline to run.",
    )
    args, unknown = parser.parse_known_args()

    module_map = {
        "paragraph": "self_reflection_llm.data_src.generate_reflection_data_paragraph",
        "code": "self_reflection_llm.data_src.generate_reflection_data_code",
        "table": "self_reflection_llm.data_src.generate_reflection_data_table",
    }

    sys.argv = [sys.argv[0], *unknown]
    runpy.run_module(module_map[args.format], run_name="__main__")


if __name__ == "__main__":
    main()
