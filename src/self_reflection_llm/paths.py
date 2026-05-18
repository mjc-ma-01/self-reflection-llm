from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_SRC_DIR = PROJECT_ROOT / "data_src"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_ANSWER_DIR = RESULTS_DIR / "model_answer"
DRA_PROCESSED_DIR = RESULTS_DIR / "DRA_processed"
RL_DATA_DIR = PROJECT_ROOT / "RL_data"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
