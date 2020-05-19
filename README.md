# PolymerStructurePredictor (PSP)

In the case of organic oligomers/polymers, preparing 3D-geometries is the first and crucial step for high-level *ab-initio* calculation. Building 3D structures by hand is not only a tedious job but sometimes challenging as well. The PSP makes your life a lot easier by automatically generating proper 3D structures from SMILES strings.

>The PSP generates monomer, oligomers and polymer chains as well as crystal structures from a SMILES string.

## Documentation, Discussion and Contribution
* PSP documents are available on [Ramprasad Group website](http://ramprasad.mse.gatech.edu/).
* Mailing list:
* Issues: 

## Installation

### Requirements
* [RDKit](https://www.rdkit.org/)
* [Open Babel](https://openbabel.org/docs/dev/index.html)

Note that, both [RDKit](https://www.rdkit.org/) and [Open Babel](https://openbabel.org/docs/dev/index.html) are need to be installed manually.  For detailed installation procedure visit toolkit’s website: [Open Babel](https://openbabel.org/docs/dev/index.html), [RDKit](https://www.rdkit.org/). Make sure that all dependencies are installed correctly and working properly.

We recommend using Anaconda python, and creating a
fresh conda environment for the install (e. g. `conda create -n MY_ENV_NAME`).

Then use the included setup.py procedure, from the cloned directory.

```angular2
python setup.py develop
```