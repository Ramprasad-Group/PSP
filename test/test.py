import pandas as pd
import PolymerStructurePredictor.PolymerBuilder as PB

df_smiles = pd.read_csv('test.csv', low_memory=False)   # fingerprinted data
df_smiles=df_smiles[['RID','smiles']]

chain_builder = PB.PolymerBuilder(df_smiles)
results = chain_builder.BuildPolymer()
print(results)