import pandas as pd
import unittest
import glob
import os
import psp.ChainBuilder as ChB
import psp.CrystalBuilder as CrB


TEST_DIR = os.path.abspath(os.path.dirname(__file__))


class PspGeneralTest(unittest.TestCase):
    def test_crystal_build(self):
        df_smiles = pd.read_csv(
            os.path.join(TEST_DIR, "chain.csv"), low_memory=False
        )  # fingerprinted data

        chain_builder = ChB.Builder(
            Dataframe=df_smiles,
            ID_col="PID",
            SMILES_col="smiles_polymer",
            Length=["n"],
            Steps=25,
            Substeps=10,
            MonomerAng="medium",
            DimerAng="medium",
            Method="SA",
            OutDir="chains",
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
            OutDir="crystals",
        )
        results = crystal_builder.BuildCrystal()
        self.assertIsNotNone(results)
        print(results)
