import pandas as pd
import psp.AmorphousBuilder as ab

input_df = pd.read_csv("input_amor.csv", low_memory=False)
amor = ab.Builder(
    input_df,
    ID_col="ID",
    SMILES_col="smiles",
    OutDir='amor_model',
    Length='Len',
    NumConf='NumConf',
    NumModel=1,
    LeftCap = "LeftCap",
    RightCap = "RightCap",
    Loop='Loop',
    density=0.85,
    box_type='c',
)
amor.Build()
