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
    Loop='Loop',
    density=0.85,
    box_type='r',
    #box_size=[0.0,20,0.0,20,0.0,20]
)
amor.Build()
