from viasegura.downloader import Downloader
from viasegura.labelers import LanesLabeler, ModelLabeler  # noqa: F401

dl = Downloader()
download_models = dl.download
