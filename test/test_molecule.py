import pandas as pd
import psp.MoleculeBuilder as mb

df_smiles = pd.read_csv("molecule.csv", low_memory=False)
mol = mb.Builder(df_smiles, ID_col="ID", SMILES_col="smiles", OutDir='molecules', Inter_Mol_Dis=6, Length=[10], NumConf=1, loop=True)
results = mol.Build3D()
print(results)
exit()
# MM or MD
from pysimm import system, lmps, forcefield
s = system.read_mol('molecules/PVC_N10_C1.mol')

# the resulting system has sufficient information to type with a forcefield, here we will use the GAFF2 force field
s.apply_forcefield(forcefield.Gaff2())

# we'll perform energy minimization using the fire algorithm in LAMMPS
lmps.quick_min(s, min_style='fire')

# write a few different file formats
s.write_xyz('methanol.xyz')
s.write_yaml('methanol.yaml')
s.write_lammps('methanol.lmps')
s.write_chemdoodle_json('methanol.json')
