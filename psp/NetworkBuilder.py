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
        OutDir='networks',
        Inter_Mol_Dis=6,
        IrrStruc=False,
        OPLS=False,
        GAFF2=False,
        GAFF2_atom_typing='pysimm',
        Subscript=False,
    ):
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.OutDir = OutDir
        self.Dataframe = Dataframe
        self.NCores = NCores
        self.Inter_Mol_Dis = Inter_Mol_Dis
        self.IrrStruc = IrrStruc
        self.OPLS = OPLS
        self.GAFF2 = GAFF2
        self.GAFF2_atom_typing = GAFF2_atom_typing
        self.Subscript = Subscript

    # list of molecules name and CORRECT/WRONG
    def Build(self):
        if self.Subscript is False:
            out_lib.print_psp_info()  # Print PSP info
        out_lib.print_input("NetworkBuilder", self.Dataframe)

        if self.OPLS is True:
            self.NCores = 1
        if self.NCores <= 0:
            ncore_print = 'All'
        else:
            ncore_print = self.NCores

        print(
            "\n",
            "Additional information: ",
            "\n",
            "Run short MD simulation: ",
            self.IrrStruc,
            "\n",
            "Generate OPLS parameter file: ",
            self.OPLS,
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
            delayed(lib.build_pn)(
                unit_name,
                df,
                self.ID_col,
                self.SMILES_col,
                self.Inter_Mol_Dis,
                self.IrrStruc,
                self.OPLS,
                self.GAFF2,
                self.GAFF2_atom_typing,
                NCores_opt,
                out_dir,
            )
            for unit_name in tqdm(
                df[self.ID_col].values, desc='Building polymer networks ...',
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
            "Polymer networks",
            np.round((end_1 - start_1) / 60, 2),
            self.Subscript,
        )
        return chk_tri
