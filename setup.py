import os
from setuptools import setup, find_packages
from subprocess import call

# Test for openbabel/rdkit conda installs
try:
    from openbabel import openbabel
except ImportError:
    raise ModuleNotFoundError("openbabel not found, install openbabel via conda-forge.")

try:
    import rdkit
except ImportError:
    raise ModuleNotFoundError("rdkit not found, install openbabel via conda-forge.")

# Get PATH for external software and write in .bashrc
HOME_DIR = os.environ.get('HOME')
# PACKMOL
if os.getenv('PACKMOL_EXEC') is None:
    print("Enter PATH for PACKMOL executable: ")
    print("For example '/home/opt/packmol/packmol'")
    packmol_exec = input("")
    call("echo \# PACKMOL_PSP >> {}".format(os.path.join(HOME_DIR,'.bashrc')),shell=True)
    call("echo export PACKMOL_EXEC={} >> {}".format(packmol_exec,os.path.join(HOME_DIR,'.bashrc')),shell=True)

# ORCA
if os.getenv('ORCA_EXEC') is None:
    print("Enter PATH for ORCA executable: ")
    print("For example '/home/opt/orca_4_2/orca'")
    orca_exec = input("")
    call("echo \# ORCA_PSP >> {}".format(os.path.join(HOME_DIR,'.bashrc')),shell=True)
    call("echo export ORCA_EXEC={} >> {}".format(orca_exec,os.path.join(HOME_DIR,'.bashrc')),shell=True)

# OPENMPI
if os.getenv('OPENMPI_bin') is None:
    print("Enter PATH for OPENMPI: ")
    print("For example '/home/opt/openmpi-316'")
    openmpi_path = input("")
    call("echo \# OPENMPI_PSP >> {}".format(os.path.join(HOME_DIR,'.bashrc')),shell=True)
    call("echo export OPENMPI_bin={} >> {}".format(os.path.join(openmpi_path,'bin'),os.path.join(HOME_DIR,'.bashrc')),shell=True)
    call("echo export OPENMPI_lib={} >> {}".format(os.path.join(openmpi_path,'lib'),os.path.join(HOME_DIR,'.bashrc')),shell=True)

# Read the contents of your README file
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(PACKAGE_DIR, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


setup(name='PolymerStructurePredictor',
      version='1.0.0',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      description='Build single chains and crystal structures of polymers',
      keywords=['SMILES', 'polymer', 'single chain', 'crystal structure'],
      url='https://github.com/Ramprasad-Group/PSP',
      author='Harikrishna Sahu',
      author_email='harikrishnasahu89@gmail.com',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
      #license='MIT',
      packages=find_packages(),
      install_requires=['scipy',
                        'pandas',
                        'joblib'],
      zip_safe=False
      )
