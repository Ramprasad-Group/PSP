#import pandas as pd
#import PSP.PolymerBuilder as PB
import PSP.CrystalBuilder as CB

#df_smiles = pd.read_csv('test.csv', low_memory=False)   # fingerprinted data
#df_smiles=df_smiles[['ID','smiles']]

#chain_builder = PB.Builder(df_smiles,ID_col='PID', SMILES_col='smiles_polymer', 
#        num_conf=1, Steps=25, Substeps=10, input_monomer_angles='medium', 
#        input_dimer_angles='medium', method='SA', n_cores=0)
#results = chain_builder.BuildPolymer()
#print('>>>', results)
#exit()

ID='P3'
import glob
try:
    list = glob.glob("Single-Chains/"+ID+'/'+"*.vasp")
    print (list)
    crystal = CB.Builder(VaspInp_list=list, Nsamples=5, Input_radius='auto', 
            OutDir='Crystals/')
    results = crystal.BuildCrystal()
    print(results)
except:
    print("Check: output/"+ID, " directory")
