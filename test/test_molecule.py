import pandas as pd
import psp.MoleculeBuilder as mb

df_smiles = pd.read_csv("molecule.csv", low_memory=False)
mol = mb.Builder(df_smiles, ID_col="ID", SMILES_col="smiles", OutDir='molecule')
results = mol.Build3D()
print(results)
