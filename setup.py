from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'VIAsegura-test'
LONG_DESCRIPTION = 'Test for package that executes iRAP label module'

setup(
       # the name must match the folder name 'verysimplemodule'
        name="VIAsegura-test", 
        version=VERSION,
        author="Jose Maria Marquez Blanco",
        author_email="jose.marquez.blanco@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['tensorflow-gpu','numpy', 'tqdm', 'opencv-python'],
        
        keywords=['Machine Learning', 'safe road'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Profesional",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)