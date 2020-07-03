from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='PolymerStructurePredictor',
version='0.1.0',
long_description=long_description,
long_description_content_type='text/markdown',
description='Build single chains and crystal structures of polymers',
keywords=['SMILES', 'polymer', 'single chain', 'crystal structure'],
url='#',
author='Harikrishna Sahu',
author_email='harikrishnasahu89@gmail.com',
license='MIT',
packages=['PSP'],
#install_requires='openbabel',
zip_safe=False)
