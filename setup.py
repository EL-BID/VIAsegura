from setuptools import setup, find_packages
import pkg_resources
from pathlib import Path

this_directory = Path(__file__).parent
VERSION = '0.1.42' 
DESCRIPTION = 'A python package to interact with Inter-American Development Bank machine learning models to automatic label elements for iRAP certification'

LONG_DESCRIPTION = (this_directory / "README.md").read_text()


with Path('requirements.txt').open() as requirements_txt:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]

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
        install_requires=install_requires,        
        keywords=['Machine Learning', 'safe road'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.7",    
        ],
        python_requires=">=3.7", 
        include_package_data=True
)