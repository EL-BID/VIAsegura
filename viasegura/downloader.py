import logging
from pathlib import Path

from viasegura import VIASEGURA_PATH

logger = logging.getLogger(__name__)


class Downloader:

    def __init__(self, models_path=VIASEGURA_PATH / "models" / "models_artifacts"):
        """
        This class allows to download de models and other model data from the Inter-American Development Bank repositories

        Parameters
        ----------

        models_path: str (default instalation path of package)
                The route where is going to download and check the artifacts of the models
        """
        self.models_path = models_path

    def check_artifacts(self):
        """
        This function allows to check if the path for downloads exists
        """
        if not Path(self.models_path).is_dir():
            raise ImportError(
                "The route for the models is not present, it means that the models are not downloaded on this " "environment."
            )

    def check_files(self, filePath):
        """
        This function allows to chec if an specific file exists

        Parameters
        ----------

        filePath: str
                Route of the file to be checked

        """
        if Path(filePath).is_file():
            return True
        else:
            return False
