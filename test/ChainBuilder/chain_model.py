import pandas as pd
import psp.ChainBuilder as ChB

df_smiles = pd.read_csv("input_chain.csv")
chain_builder = ChB.Builder(
    Dataframe=df_smiles,
    ID_col="PID",
    SMILES_col="smiles_polymer",
    NumConf=1,
    Length=["n", 5],
    Steps=100,
    Substeps=20,
    Method="SA",
    NCores=1,
    OutDir='chains',
    Tol_ChainCorr=50,
    Inter_Chain_Dis=12,
)
results = chain_builder.BuildChain()
