import pandas as pd
import psp.ChainBuilder as ChB

df_smiles = pd.read_csv("chain.csv", low_memory=False)  # fingerprinted data
chain_builder = ChB.Builder(
    Dataframe=df_smiles,
    ID_col="PID",
    SMILES_col="smiles_polymer",
    NumConf=1,
    Length=["n", 5],
    Steps=20,
    Substeps=20,
    MonomerAng="intense",
    DimerAng="intense",
    Method="SA",
    NCores=0,
    OutDir='chains',
    Tol_ChainCorr=50,
)
results = chain_builder.BuildChain()
print(results)
