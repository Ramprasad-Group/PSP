# Welcome to PolymerStructurePredictor (PSP)

PSP documents are available on [Ramprasad Group website](http://ramprasad.mse.gatech.edu/).

## Installation

### Requirements
* [RDKit](https://www.rdkit.org/)
* [Open Babel](https://openbabel.org/docs/dev/index.html)

Note that, both [RDKit](https://www.rdkit.org/) and [Open Babel](https://openbabel.org/docs/dev/index.html) are need to be installed manually.  For detailed installation procedure visit toolkit’s website [Open Babel](https://openbabel.org/docs/dev/index.html), [RDKit](https://www.rdkit.org/). Make sure that all dependencies are installed correctly and working properly.

Please install rdkit and graphviz first, because they are not pip installable.
currently rdkit is not pip installable

We recommend using Anaconda python, and creating a
fresh conda environment for the install (e. g. `conda create -n MY_ENV_NAME`).

Then use the included setup.py procedure, from the cloned directory.

```angular2
python setup.py develop
```