from viasegura.labelers import Preprocess, ModelLabeler, LanesLabeler
from viasegura.downloader import Downloader
import os 
import sys

dl = Downloader()
download_models = dl.download

