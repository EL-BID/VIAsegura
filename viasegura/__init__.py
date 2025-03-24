import logging
import os
from pathlib import Path

VIASEGURA_PATH = Path(__file__).parent

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # for keras >= 3.0.0

from viasegura.configs.config import setup_logging  # noqa: E402

setup_logging(level=logging.INFO)

from viasegura.labelers import LanesLabeler, ModelLabeler  # noqa: F401, E402
