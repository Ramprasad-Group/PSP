import pandas as pd
import glob
import os
import psp.ChainBuilder as ChB
import psp.CrystalBuilder as CrB

df_smiles = pd.read_csv("chain.csv", low_memory=False)  # fingerprinted data

chain_builder = ChB.Builder(
    df_smiles,
    ID_col="PID",
    SMILES_col="smiles_polymer",
    length=["n"],
    Steps=25,
    Substeps=10,
    input_monomer_angles="medium",
    input_dimer_angles="medium",
    method="SA",
)
results = chain_builder.BuildChain()
print(results)
ID = "PVC2"
vasp_input_list = glob.glob("chains/" + ID + "/" + "*.vasp")
crystal_builder = CrB.Builder(
    VaspInp_list=vasp_input_list,
    Nsamples=5,
    Input_radius="auto",
    OutDir="crystals/",
)
results = crystal_builder.BuildCrystal()
print(results)
