import pandas as pd
import PolymerStructurePredictor.PolymerBuilder as PB
import PolymerStructurePredictor.CrystalBuilder as CB

df_smiles = pd.read_csv('test.csv', low_memory=False)   # fingerprinted data
df_smiles=df_smiles[['RID','smiles']]

chain_builder = PB.PolymerBuilder(df_smiles,length=[1,2,3,'n'],n_cores=0,)
results = chain_builder.BuildPolymer()

print(results)

list=['123/PE', '456/R1004531_1']
crystal = CB.CrystalBuilder(VaspInp_list=list, Nsamples=3, Input_radius='auto', OutDir='Crystals/')
result=crystal.build()