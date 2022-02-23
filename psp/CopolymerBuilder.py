import numpy as np
import pandas as pd
import psp.PSP2_lib as lib
import psp.PSP_lib as bd
import os
import shutil
import time
import multiprocessing
from joblib import Parallel, delayed
import psp.output_lib as out_lib
from tqdm import tqdm


class Builder:
    def __init__(
        self,
        Dataframe,
        NCores=0,
        ID_col='ID',
        SMILES_col='smiles',
        LeftCapSMI_col='LeftCap',
        RightCapSMI_col='RightCap',
        Nunits_col='Nunits',  # list of numbers (ratios); follow the order of building blocks
        Tunits_col='Tunits',  # Total number units in a polymer chain
        Mwt_col='Mwt_polymer',  # if > 0, then Nunits will be determined from molar mass of BB and Mwt
        Copoly_type_col='Copoly_type',  # 'homo', # homo, alternating, block, graft, random
        define_BB_col='define_BB',
        Loop_col='Loop',
        OutDir='copolymer_models',
        NumConf=1,
        Inter_Mol_Dis=6,
        Output=['xyz'],
        IrrStruc=False,
        GAFF2_atom_typing='pysimm',
        Subscript=False,
        seed=None,
        rdkit_conf_param={},
    ):
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.LeftCapSMI_col = LeftCapSMI_col
        self.RightCapSMI_col = RightCapSMI_col
        self.Nunits_col = Nunits_col
        self.Tunits_col = Tunits_col
        self.Mwt_col = Mwt_col
        self.Copoly_type_col = Copoly_type_col
        self.define_BB_col = define_BB_col
        self.OutDir = OutDir
        self.Dataframe = Dataframe
        self.NCores = NCores
        self.NumConf = NumConf
        self.Inter_Mol_Dis = Inter_Mol_Dis
        self.Output = Output
        self.Loop_col = Loop_col
        self.IrrStruc = IrrStruc
        self.GAFF2_atom_typing = GAFF2_atom_typing
        self.Subscript = Subscript
        self.seed = (seed,)
        self.rdkit_conf_param = (rdkit_conf_param,)

    # list of molecules name and CORRECT/WRONG
    def Build(self):
        if self.Subscript is False:
            out_lib.print_psp_info()  # Print PSP info
        out_lib.print_input("NetworkBuilder", self.Dataframe)

        OPLS_list = ['OPLS', 'OPLS-AA', 'opls', 'opls-aa']
        if any(i in OPLS_list for i in self.Output):
            self.NCores = 1
        if self.NCores <= 0:
            ncore_print = 'All'
        else:
            ncore_print = self.NCores

        print(
            "\n",
            "Additional information: ",
            "\n",
            "Output files: ",
            self.Output,
            "\n",
            "Run short MD simulation: ",
            self.IrrStruc,
            "\n",
            "Intermolecular distance in POSCAR: ",
            self.Inter_Mol_Dis,
            "\n",
            "Number of cores: ",
            ncore_print,
            "\n",
            "Output Directory: ",
            self.OutDir,
            "\n",
            "Random seed: ",
            self.seed,
            "\n",
        )

        # location of directory for VASP inputs (polymers) and build a directory
        out_dir = self.OutDir + '/'
        bd.build_dir(out_dir)

        # Directories
        # Working directory
        bd.build_dir('work_dir/')

        start_1 = time.time()
        list_out_xyz = 'output_NB.csv'
        chk_tri = []

        df = self.Dataframe.copy()
        df[self.ID_col] = df[self.ID_col].apply(str)

        if self.NCores == 0:
            self.NCores = multiprocessing.cpu_count() - 1

        if self.NCores == -1 or self.IrrStruc is True:
            NCores_opt = 0
            self.NCores = 1
        else:
            NCores_opt = 1

        result = Parallel(n_jobs=self.NCores)(
            delayed(lib.build_copoly)(
                unit_name,
                df,
                self.ID_col,
                self.SMILES_col,
                self.LeftCapSMI_col,
                self.RightCapSMI_col,
                self.Nunits_col,
                self.Tunits_col,
                self.Mwt_col,
                self.Copoly_type_col,
                self.define_BB_col,
                self.NumConf,
                self.Inter_Mol_Dis,
                self.Output,
                self.Loop_col,
                self.IrrStruc,
                self.GAFF2_atom_typing,
                NCores_opt,
                out_dir,
                self.seed,
                self.rdkit_conf_param,
            )
            for unit_name in tqdm(
                df[self.ID_col].values, desc='Building copolymers ...',
            )
        )

        for i in result:
            chk_tri.append([i[0], i[1]])

        chk_tri = pd.DataFrame(chk_tri, columns=['ID', 'Result'])
        chk_tri.to_csv(list_out_xyz)

        bd.del_tmp_files()

        # Delete work directory
        if os.path.isdir('work_dir/'):
            shutil.rmtree('work_dir/')

        end_1 = time.time()
        out_lib.print_out(
            chk_tri,
            "Copolymer models",
            np.round((end_1 - start_1) / 60, 2),
            self.Subscript,
        )
        return chk_tri
