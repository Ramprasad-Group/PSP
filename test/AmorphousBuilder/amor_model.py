import pandas as pd
import psp.AmorphousBuilder as ab

input_df = pd.read_csv("input_amor.csv", low_memory=False)
amor = ab.Builder(
    input_df,
    ID_col="ID",
    SMILES_col="smiles",
    OutDir='amorphous_models',
    Length='Len',
    NumConf='NumConf',
    density=0.65,
    loop=False,
)
amor.Build()
