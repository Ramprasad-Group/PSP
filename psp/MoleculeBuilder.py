import numpy as np
import pandas as pd
import psp.PSP_lib as bd
import os
import shutil
import time
import multiprocessing
from joblib import Parallel, delayed
import psp.output_lib as lib
from tqdm import tqdm


class Builder:
    def __init__(
        self,
        Dataframe,
        NCores=0,
        ID_col='ID',
        SMILES_col='smiles',
        LeftCap='LeftCap',
        RightCap='RightCap',
        OutDir='molecules',
        Inter_Mol_Dis=6,
        Length=[1],
        NumConf=1,
        Loop=False,
        IrrStruc=False,
        OPLS=False,
        GAFF2=False,
        GAFF2_atom_typing='pysimm',
        Subscript=False,
    ):
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.LeftCap = LeftCap
        self.RightCap = RightCap
        self.OutDir = OutDir
        self.Dataframe = Dataframe
        self.NCores = NCores
        self.Inter_Mol_Dis = Inter_Mol_Dis
        self.Length = Length
        self.NumConf = NumConf
        self.Loop = Loop
        self.IrrStruc = IrrStruc
        self.OPLS = OPLS
        self.GAFF2 = GAFF2
        self.GAFF2_atom_typing = GAFF2_atom_typing
        self.Subscript = Subscript

    # list of molecules name and CORRECT/WRONG
    def Build(self):
        if self.Subscript is False:
            lib.print_psp_info()  # Print PSP info
        lib.print_input("MoleculeBuilder", self.Dataframe)
        if self.NCores <= 0:
            ncore_print = 'All'
        else:
            ncore_print = self.NCores

        print(
            "\n",
            "Additional information: ",
            "\n",
            "Length of oligomers: ",
            self.Length,
            "\n",
            "Number of conformers: ",
            self.NumConf,
            "\n",
            "Loop model: ",
            self.Loop,
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

        # location of input XYZ files
        xyz_in_dir = 'work_dir/xyz-in/'
        bd.build_dir(xyz_in_dir)

        start_1 = time.time()
        list_out_xyz = 'output_MB.csv'
        chk_tri = []
        # ID =
        # SMILES = self.SMILES_col
        df = self.Dataframe.copy()
        df[self.ID_col] = df[self.ID_col].apply(str)

        if self.NCores == 0:
            self.NCores = multiprocessing.cpu_count() - 1

        if self.NCores == -1 or self.IrrStruc is True:
            NCores_opt = 0
            self.NCores = 1
        else:
            NCores_opt = 1
        print("\n 3D model building started...\n")
        result = Parallel(n_jobs=self.NCores)(
            delayed(bd.build_3D)(
                unit_name,
                df,
                self.ID_col,
                self.SMILES_col,
                self.LeftCap,
                self.RightCap,
                out_dir,
                self.Inter_Mol_Dis,
                self.Length,
                xyz_in_dir,
                self.NumConf,
                self.Loop,
                self.IrrStruc,
                self.OPLS,
                self.GAFF2,
                self.GAFF2_atom_typing,
                NCores_opt,
            )
            for unit_name in tqdm(
                df[self.ID_col].values, desc='Building models ...',
            )
        )
        # print(result)
        # exit()
        for i in result:
            chk_tri.append([i[0], i[1], i[2]])

        chk_tri = pd.DataFrame(chk_tri, columns=['ID', 'Result', 'SMILES'])
        chk_tri.to_csv(list_out_xyz)

        bd.del_tmp_files()

        # Delete work directory
        if os.path.isdir('work_dir/'):
            shutil.rmtree('work_dir/')

        end_1 = time.time()
        lib.print_out(
            chk_tri, "3D model", np.round((end_1 - start_1) / 60, 2), self.Subscript
        )
        return chk_tri
