import pandas as pd
import psp.AmorphousBuilder as ab

input_df = pd.read_csv("input_amor.csv", low_memory=False)
amor = ab.Builder(
    input_df,
    ID_col="ID",
    SMILES_col="smiles",
    Length='Len',
    NumConf='NumConf',
    LeftCap = "LeftCap",
    RightCap = "RightCap",
    Loop='Loop',
    density=0.85,
    box_type='c',
    BondInfo=False
)
amor.Build()
amor.get_opls(output_fname='amor_opls.lmps')
