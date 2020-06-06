import pandas as pd
import PolymerStructurePredictor.PolymerBuilder as PB
import PolymerStructurePredictor.CrystalBuilder as CB

df_smiles = pd.read_csv('test.csv', low_memory=False)   # fingerprinted data
df_smiles=df_smiles[['RID','smiles']]

chain_builder = PB.PolymerBuilder(df_smiles,num_conf=1, length=[1,2,'n'], Steps=100, Substeps=20, input_monomer_angles='intense', input_dimer_angles='medium', n_cores=0)
results = chain_builder.BuildPolymer()
print(results)


ID='P100801'
import glob
try:
    list = glob.glob("output/"+ID+'/'+"*.vasp")
    crystal = CB.CrystalBuilder(VaspInp_list=list, Nsamples=3, Input_radius='auto', OutDir='Crystals/')
    results = crystal.build()
    print(results)
except:
    print("Check: output/"+ID, " directory")