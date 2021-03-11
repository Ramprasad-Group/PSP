import pandas as pd
import psp.DimerBuilder as db

input_data = pd.read_csv('input_DM.csv')
dim = db.Builder(
    input_data,
    ID_col="ID",
    SMILES_col="smiles",
    OutDir='dimer-xyz',
    Ndimer=10,
    ABdis=2.0,
)
results = dim.Build()
