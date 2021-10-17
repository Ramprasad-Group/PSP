import pandas as pd
import glob
import psp.ChainBuilder as ChB
import psp.CrystalBuilder as CrB

df_smiles = pd.read_csv("input_chain.csv", low_memory=False)  # fingerprinted data

chain_builder = ChB.Builder(
    Dataframe=df_smiles,
    ID_col="PID",
    SMILES_col="smiles_polymer",
    NumConf=1,
    Length=['n',5],
    Steps=20,
    Substeps=20,
    Method="SA",
    NCores=1,
    OutDir='chains',
    Tol_ChainCorr=50,
)
results = chain_builder.BuildChain()

ID = "PVC"
vasp_input_list = glob.glob("chains/" + ID + "/" + "*.vasp")
crystal_builder = CrB.Builder(
    VaspInp_list=vasp_input_list,
    NSamples=10,
    InputRadius="auto",
    MinAtomicDis=2.0,
    Polymer=True,
    Optimize=False,
    NCores=1,
)
results = crystal_builder.BuildCrystal()
