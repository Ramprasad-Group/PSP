import pandas as pd
import psp.AmorphousBuilder as ab

input_df = pd.read_csv("input_PE.csv", low_memory=False)
amor = ab.Builder(
    input_df,
    ID_col="ID",
    SMILES_col="smiles",
    OutDir='PE_tests',
    Length='Len',
    NumConf='NumConf',
    LeftCap = "LeftCap",
    RightCap = "RightCap",
    Loop='Loop',
    density=0.85,
    box_type='c',
    BondInfo=False
    #box_size=[0.0,20,0.0,20,0.0,20]
)
amor.Build()
amor.get_opls(output_fname='amor_opls.lmps')
