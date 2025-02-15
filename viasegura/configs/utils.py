import json
import logging
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent


def setup_logging(level: int = logging.DEBUG) -> None:
    """Sets up the logging.

    The logging level can be set to change the verbosity of the output.

    Args:
        level: The logging level to use. Defaults to logging.DEBUG.

    Returns:
        None
    """
    format = (
        "%(asctime)s - %(levelname)s - %(message)s" if level == logging.INFO else "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    logging.basicConfig(
        level=level,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def update(d: dict, u: dict) -> dict:
    """Recursively updates a dictionary.

    The update dictionary is merged into the target dictionary.

    Args:
        d: The target dictionary.
        u: The update dictionary.

    Returns:
        The updated target dictionary.
    """
    for k, v in u.items():
        if isinstance(d, dict):
            if isinstance(v, dict):
                r = update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}
    return d


class Config_Basic:
    """Basic configuration class.

    Attributes:
        config (dict): The loaded configuration.
    """

    def __init__(self):
        """Initializes the configuration class.

        Args:
            config_file (Path): The path to the configuration file.
            config_file_update (Path): The path to the updated configuration file.
        """
        self.config = None

    def load_config(self, config_file: Path, config_file_update: Path = None) -> None:
        """Loads a configuration from the specified file.

        Args:
            config_file (Path): The path to the configuration file.
            config_file_update (Path): The path to the updated configuration file.

        Returns:
            None
        """
        with open(str(config_file), "r") as f:
            self.config = json.loads(f.read())

        if config_file_update:
            with open(str(config_file_update), "r") as f:
                self.config = update(self.config, json.loads(f.read()))
