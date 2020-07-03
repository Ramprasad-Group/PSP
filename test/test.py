import pandas as pd
import PSP.PolymerBuilder as PB
import PSP.CrystalBuilder as CB

df_smiles = pd.read_csv('test.csv', low_memory=False)   # fingerprinted data
#df_smiles=df_smiles[['ID','smiles']]

chain_builder = PB.Builder(df_smiles,ID_col='PID',SMILES_col='smiles_polymer', num_conf=1, Steps=50, Substeps=10, input_monomer_angles='low', input_dimer_angles='low', method='SA', n_cores=1)
results = chain_builder.BuildPolymer()
print(results)
#exit()
ID='P1'
import glob
try:
    list = glob.glob("Single-Chains/"+ID+'/'+"*.vasp")
    crystal = CB.Builder(VaspInp_list=list, Nsamples=5, Input_radius='auto', OutDir='Crystals/')
    results = crystal.BuildCrystal()
    print(results)
except:
    print("Check: output/"+ID, " directory")