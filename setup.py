from os import path
from setuptools import setup, find_packages

# Test for openbabel/rdkit conda installs
try:
    from openbabel import openbabel
except ImportError:
    raise ModuleNotFoundError("openbabel not found, install openbabel via conda-forge.")

try:
    import rdkit
except ImportError:
    raise ModuleNotFoundError("rdkit not found, install openbabel via conda-forge.")


# Read the contents of your README file
PACKAGE_DIR = path.abspath(path.dirname(__file__))
with open(path.join(PACKAGE_DIR, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


setup(name='PolymerStructurePredictor',
      version='1.0.0',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      description='Build single chains and crystal structures of polymers',
      keywords=['SMILES', 'polymer', 'single chain', 'crystal structure'],
      # TODO: add github repo url here
      url='#',
      author='Harikrishna Sahu',
      author_email='harikrishnasahu89@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['scipy',
                        'pandas',
                        'joblib'],
      zip_safe=False
      )
