import pandas as pd
import psp.MoleculeBuilder as mb

df_smiles = pd.read_csv("molecule.csv", low_memory=False)
mol = mb.Builder(df_smiles, ID_col="ID", SMILES_col="smiles", OutDir='molecules', Inter_Mol_Dis=6, Length=[1,5], NumConf=1)
results = mol.Build3D()
print(results)
