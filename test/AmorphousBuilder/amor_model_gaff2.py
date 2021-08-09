import pandas as pd
import psp.AmorphousBuilder as ab

input_df = pd.read_csv("input_amor1.csv", low_memory=False)
amor = ab.Builder(
    input_df,
    ID_col="ID",
    SMILES_col="smiles",
    Length='Len',
    NumConf='NumConf',
    LeftCap = "LeftCap",
    RightCap = "RightCap",
    Loop='Loop',
    density=0.85,
    box_type='c',
    BondInfo=False
)
amor.Build()

# Default get_gaff2() uses Pysimm for atom typing
amor.get_gaff2(output_fname='amor_gaff2.lmps')

'''
[ADVANCED] If Ambertools is installed, and antechamber is in the PATH
(e.g. export ANTECHAMBER_EXEC=~/.conda/envs/AmberTools21/bin/antechamber),
atom typing can also be done using antechamber by specifying atom_typing='antechamber'.
In addition, atom types can be swapped manually by specifying the swap_dict
(e.g. swap_dict={'ns': 'n'}).

A representative example is as below:
amor.get_gaff2(output_fname='amor_gaff2.lmps', atom_typing='antechamber', swap_dict={'ns': 'n'})
'''
