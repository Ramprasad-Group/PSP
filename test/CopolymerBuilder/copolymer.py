import pandas as pd
import psp.CopolymerBuilder as cob

df_smiles = pd.read_csv("alternating.csv")

mol = cob.Builder(
    df_smiles,
    ID_col="ID",
    SMILES_col="smiles",
    LeftCap='LeftCap',
    RightCap='RightCap',
    Nunits_col='Nunits',
    Mwt_col=0,
    Copoly_type_col='CopolymerType',
    define_BB_col='define_BB',
    OutDir='copolymer_models',
    NCores=1,
)
results = mol.Build()