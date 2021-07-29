# PolymerStructurePredictor (PSP)

In the case of organic oligomers/polymers, preparing 3D-geometries is the first and crucial step for high-level *ab-initio* calculations. Building 3D structures by hand is not only a tedious job but sometimes challenging as well. The PSP makes your life a lot easier by automatically generating proper 3D structures from SMILES strings.

>The PSP generates models for monomers, oligomers, infinite polymer chains, crystal  as well as amorphous structures from a SMILES string.

## Contributors

## License & copyright
Ramprasad Group, Georgia Tech, USA

Licensed under the [MIT License](LICENSE). 

## Documentation, Discussion and Contribution
* PSP documents are available on [Ramprasad Group website](http://ramprasad.mse.gatech.edu/).
* Mailing list:
* Issues: 

## Installation

### Requirements
* [RDKit](https://www.rdkit.org/) v2020.09.1.0
* [Open Babel](https://open-babel.readthedocs.io/en/latest/index.html) v3.1.1
* [PACKMOL](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml) v20.2.2
* [PySIMM](https://pysimm.org/) v0.2.3
* [LAMMPS](https://docs.lammps.org/Manual.html)
* [LigParGen](http://zarbi.chem.yale.edu/ligpargen/) v2.1

It should be noted that all dependencies must be installed separately and tested to ensure that they all function. We recommend using Anaconda python, and creating a fresh conda environment for the install (e. g. `conda create -n MY_ENV_NAME`).

RDKit and OpenBabel are available as conda packages and can be installed as mentioned in the following links (1)[https://anaconda.org/rdkit/rdkit](https://anaconda.org/rdkit/rdkit) and (2)[https://anaconda.org/conda-forge/openbabel](https://anaconda.org/conda-forge/openbabel).

The installation procedure for the PACKMOL package may be found at the following link: [http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml). Make sure to include the path for PACKMOL executable as an environment variable "PACKMOL\_EXEC".

LAMMPS can be installed separately or along with PySIMM. Make sure to add the PySIMM package to your PYTHONPATH and add PySIMM and LAMMPS command-line tools to your PATH as mentioned in the PySIMM documentation.

Following that, source your ~/.bashrc file.  PSP will look for PATHs for PACKMOL, PySIMM, and LAMMPS while performing its tasks.

PSP requires LigParGen 2.1 installed in a local server that needs BOSS as well as the following packages (1) Openbabel 2.4, (2) Network 1.11, (3) pandas 0.25.1. The following link has a detailed description of the installation: [http://zarbi.chem.yale.edu/ligpargen](http://zarbi.chem.yale.edu/ligpargen/). The user either needs to modify the LigParGen source codes or replace them with the provided ones so that the output files can be stored in the desired directory instead of the default \tmp directory. The modified source code is available at www.

Then use the included setup.py procedure, from the cloned directory.

```angular2
python setup.py develop
```
