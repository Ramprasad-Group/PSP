import pandas as pd
import glob
import os
import psp.ChainBuilder as ChB
import psp.CrystalBuilder as CrB

df_smiles = pd.read_csv("chain.csv", low_memory=False)  # fingerprinted data

chain_builder = ChB.Builder(
    Dataframe=df_smiles,
    ID_col="PID",
    SMILES_col="smiles_polymer",
    NumConf=1,
    Length=["n"],
    Steps=20,
    Substeps=20,
    MonomerAng="intense",
    DimerAng="intense",
    Method="SA",
    NCores=0,
    OutDir='chains',
    Tol_ChainCorr=50
)
results = chain_builder.BuildChain()
print(results)

ID = "PVC2"
vasp_input_list = glob.glob("chains/" + ID + "/" + "*.vasp")
crystal_builder = CrB.Builder(
    VaspInp_list=vasp_input_list,
    NSamples=5,
    InputRadius="auto",
    MinAtomicDis=2.0,
    NCores=0,
)
results = crystal_builder.BuildCrystal()
print(results)
