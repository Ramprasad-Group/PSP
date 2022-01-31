import pandas as pd
import psp.AmorphousBuilder2 as ab

input_df = pd.read_csv("alternating.csv")
amor = ab.Builder(
    input_df,
    ID_col="ID",
    SMILES_col="smiles",
    OutDir='amor_model',
    rdkit_conf_param={'numConfs':2, 'numThreads':0, 'randomSeed':2, 'ExpTorsionAnglePref': False, 'BasicKnowledge': False},
    NumModel=1,
    LeftCap = "LeftCap",
    RightCap = "RightCap",
    density=0.65,
    box_type='c',
)
amor.Build()
