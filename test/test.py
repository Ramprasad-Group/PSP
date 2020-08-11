import pandas as pd
import unittest
import glob
import os
import psp.PolymerBuilder as PB
import psp.CrystalBuilder as CB


TEST_DIR = os.path.abspath(os.path.dirname(__file__))


class PspGeneralTest(unittest.TestCase):
    def test_crystal_build(self):
        df_smiles = pd.read_csv(
            os.path.join(TEST_DIR, "chain.csv"), low_memory=False
        )  # fingerprinted data

        chain_builder = PB.Builder(
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
        results = chain_builder.BuildPolymer()
        print(results)
        ID = "PE"
        vasp_input_list = glob.glob("chains/" + ID + "/" + "*.vasp")
        crystal = CB.polymer_crystal(
            VaspInp_list=vasp_input_list,
            Nsamples=5,
            Input_radius="auto",
            OutDir="crystals/",
        )
        results = crystal.build_model()
        self.assertIsNotNone(results)
        print(results)
