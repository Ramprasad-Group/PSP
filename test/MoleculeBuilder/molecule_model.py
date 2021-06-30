import pandas as pd
import psp.MoleculeBuilder as mb

df_smiles = pd.read_csv("test.csv", low_memory=False)
mol = mb.Builder(
    df_smiles,
    ID_col="ID",
    SMILES_col="smiles",
    LeftCap = "LeftCap",
    RightCap ='RightCap',
    OutDir='test_models',
    Inter_Mol_Dis=6,
    Length=[16],#16
    NumConf=1,
    Loop=True,
    NCores=-1,
    IrrStruc=True
)
results = mol.Build()
print(results)
