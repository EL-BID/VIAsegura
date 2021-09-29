from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
VERSION = '0.0.1.26' 
DESCRIPTION = 'A python package to interact with Inter-American Development Bank machine learning modelsto automatic label elements for iRAP certification'

LONG_DESCRIPTION = (this_directory / "README.md").read_text()




setup(
       # the name must match the folder name 'verysimplemodule'
        name="viasegura", 
        version=VERSION,
        author="Jose Maria Marquez Blanco",
        author_email="jose.marquez.blanco@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=['tensorflow-gpu==2.5.0','numpy==1.18.5', 'tqdm', 'opencv-contrib-python==4.2.0.34','boto3==1.14.37'],        
        keywords=['Machine Learning', 'safe road'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        include_package_data=True
)