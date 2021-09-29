from viasegura_test.labelers import Preprocess, ModelLabeler, LanesLabeler
from viasegura_test.downloader import Downloader
import os 
import sys

dl = Downloader()
download_models = dl.download

