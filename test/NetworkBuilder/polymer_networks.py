import pandas as pd
import psp.NetworkBuilder as nb

df_smiles = pd.read_csv("test1.csv")

mol = nb.Builder(
    df_smiles,
    ID_col="ID",
    SMILES_col="smi",
    OutDir='model_networks',
    NCores=2,
)
results = mol.Build()

