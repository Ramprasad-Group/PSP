# PolymerStructurePredictor (PSP)

Three-dimensional atomic-level models of polymers are necessary prerequisites for physics-based simulation studies. Polymer structure predictor (PSP) is capable of generating a hierarchy of polymer models, ranging from oligomers to infinite chains to crystals to amorphous models, using a simplified molecular-input line-entry system (SMILES) string of the polymer repeat unit as the primary input. The output structures and accompanying force field (GAFF2/OPLS-AA) parameter files are provided for downstream DFT and MD simulations.

>PSP generates models for monomers, linear and loop oligomers, infinite polymer chains, crystal, and amorphous structures using SMILES strings.

## Contributors
* Harikrishna Sahu
* Huan Tran
* Kuan-Hsuan Shen
* Joseph Montoya
* Rampi Ramprasad

## License & copyright
Ramprasad Group, Georgia Tech, USA\
[Ramprasad Group website](http://ramprasad.mse.gatech.edu/)\
Licensed under the [MIT License](LICENSE).

## Installation
PSP requires the following packages to be installed in order to function properly:
* [RDKit](https://www.rdkit.org/) v2020.09.1.0
* [Open Babel](https://open-babel.readthedocs.io/en/latest/index.html) v3.1.1
* [PACKMOL](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml) v20.2.2
* [PySIMM](https://pysimm.org/) v0.2.3
* [LAMMPS](https://docs.lammps.org/Manual.html)
* [Ambertools](https://ambermd.org/AmberTools.php) v3.1.1
* LigParGen dependencies[](http://zarbi.chem.yale.edu/ligpargen/)

It should be noted that all dependencies must be installed separately and tested to ensure that they all function. We recommend using Anaconda python, and creating a fresh conda environment for PSP (e. g. `conda create -n MY_ENV_NAME`).

RDKit and OpenBabel are available as conda packages and can be installed using the instructions provided in the following links (1)[https://anaconda.org/rdkit/rdkit](https://anaconda.org/rdkit/rdkit) and (2)[https://anaconda.org/conda-forge/openbabel](https://anaconda.org/conda-forge/openbabel).

The deatiled intructions for the installation of PACKMOL package can be found at the following URL: [http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml). Make sure to include the path for PACKMOL executable as an environment variable "PACKMOL\_EXEC" in ~/.bashrc file.

LAMMPS can be installed separately or along with PySIMM. Make sure to add the PySIMM package to your PYTHONPATH and add PySIMM and LAMMPS command-line tools to your PATH as mentioned in the PySIMM documentation.

Ambertools is available as a conda package and can be installed using the instructions provided in the following links: [https://ambermd.org/AmberTools.php](https://ambermd.org/AmberTools.php). Make sure to include the path for the antechamber executable as an environment variable "ANTECHAMBER\_EXEC" in ~/.bashrc file.

Following that, source your ~/.bashrc file.  PSP will look for PATHs for PACKMOL, PySIMM, and LAMMPS while performing its tasks.

LigParGen and its dependencies: LigParGen requires the BOSS executable. Obtain a copy of it and set $BOSSdir variable in bash. For more information, see [http://zarbi.chem.yale.edu/ligpargen](http://zarbi.chem.yale.edu/ligpargen) and [http://zarbi.chem.yale.edu/software.html](http://zarbi.chem.yale.edu/software.html). To make LigParGen compatible with PSP, we updated it to include the following features: (1) the ability to store the output files in a user-defined directory; and (2) compatibility with the recent versions of Open Babel (v3.1.1), NetworkX (v2.5), and pandas (v1.2.4). Take note that we have not yet installed NetworkX; ensure that this is done. The updated LigParGen source code is redistributed as part of the PSP package. 

Once all dependencies are installed, clone the PSP repository and install it using the *setup.py* included in the package.

```angular2
python setup.py install
```
>**NOTE**: A colab notebook that demonstrates the step-by-step installation procedure and installs PSP and its dependencies has been provided. 

