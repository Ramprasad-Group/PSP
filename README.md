# PolymerStructurePredictor (PSP)

PSP is an autonomous model generator that can build a hierarchy of polymer models starting from oligomer/infinite polymer chains to sophisticated amorphous models, using a simplified molecular-input line-entry system (SMILES) of polymers. This toolkit allows users to tune several parameters to manage the quality of models and computational cost and produce an OPLS parameter file if desired. Models can be utilized as the starting geometry for physics-based simulations, allowing for complete automation in polymer discovery. 

>PSP generates models for monomers, linear and loop oligomers, infinite polymer chains, crystal, and amorphous structures from a SMILES string.

## Contributors
* Harikrishna Sahu
* Huan Tran
* Kuan-Hsuan Shen
* Joseph Montoya
* Rampi Ramprasad

## License & copyright
Ramprasad Group, Georgia Tech, USA

Licensed under the [MIT License](LICENSE). 

## Documentation, Discussion and Contribution
* PSP documents are available on [Ramprasad Group website](http://ramprasad.mse.gatech.edu/).
* Mailing list:
* Issues: 

## Installation
PSP requires the following packages to be installed in order to function properly:
* [RDKit](https://www.rdkit.org/) v2020.09.1.0
* [Open Babel](https://open-babel.readthedocs.io/en/latest/index.html) v3.1.1
* [PACKMOL](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml) v20.2.2
* [PySIMM](https://pysimm.org/) v0.2.3
* [LAMMPS](https://docs.lammps.org/Manual.html)
* [LigParGen](http://zarbi.chem.yale.edu/ligpargen/) v2.1

It should be noted that all dependencies must be installed separately and tested to ensure that they all function. We recommend using Anaconda python, and creating a fresh conda environment for PSP (e. g. `conda create -n MY_ENV_NAME`).

RDKit and OpenBabel are available as conda packages and can be installed using the instructions provided in the following links (1)[https://anaconda.org/rdkit/rdkit](https://anaconda.org/rdkit/rdkit) and (2)[https://anaconda.org/conda-forge/openbabel](https://anaconda.org/conda-forge/openbabel).

The deatiled intructions for the installation of PACKMOL package can be found at the following URL: [http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml). Make sure to include the path for PACKMOL executable as an environment variable "PACKMOL\_EXEC" in ~/.bashrc file.

LAMMPS can be installed separately or along with PySIMM. Make sure to add the PySIMM package to your PYTHONPATH and add PySIMM and LAMMPS command-line tools to your PATH as mentioned in the PySIMM documentation.

Following that, source your ~/.bashrc file.  PSP will look for PATHs for PACKMOL, PySIMM, and LAMMPS while performing its tasks.

PSP requires LigParGen 2.1 installed in a local server that needs BOSS as well as the following packages (1) Openbabel 2.4, (2) Network 1.11, (3) pandas 0.25.1. The following link has a detailed description of the installation: [http://zarbi.chem.yale.edu/ligpargen](http://zarbi.chem.yale.edu/ligpargen/). The user either needs to modify the LigParGen source codes or replace them with the supplied ones to store the output files in the desired directory instead of the default \tmp directory. The modified source code is available at www.

Once all dependencies are installed, clone the PSP repository and install it using the *setup.py* included in the package.

```angular2
python setup.py install
```
