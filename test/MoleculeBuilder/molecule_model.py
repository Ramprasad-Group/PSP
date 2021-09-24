import pandas as pd
import psp.MoleculeBuilder as mb

df_smiles = pd.read_csv("test.csv", low_memory=False)
mol = mb.Builder(
    df_smiles,
    ID_col="ID",
    SMILES_col="smiles",
    LeftCap = "LeftCap",
    RightCap ='RightCap',
    OutDir='models',
    Inter_Mol_Dis=6,
    Length=[1,3],#16
    NumConf=1,
    Loop=False,
    NCores=-1,
    IrrStruc=True,
    OPLS=False
)
results = mol.Build()
#print(results)
