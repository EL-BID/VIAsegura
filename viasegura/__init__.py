import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # for keras >= 3.0.0

from viasegura.downloader import Downloader
from viasegura.labelers import LanesLabeler, ModelLabeler  # noqa: F401

dl = Downloader()
download_models = dl.download
