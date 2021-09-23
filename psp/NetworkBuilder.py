import numpy as np
import pandas as pd
import psp.PSP2_lib as lib
import os
import shutil
import time
import multiprocessing
from joblib import Parallel, delayed


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

    # list of molecules name and CORRECT/WRONG
    def Build(self):

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

        result = Parallel(n_jobs=self.NCores)(
            delayed(lib.build_network)(
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
                NCores_opt,
            )
            for unit_name in df[self.ID_col].values
        )
        # print(result)
        # exit()
        for i in result:
            chk_tri.append([i[0], i[1], i[2]])

        end_1 = time.time()
        print("")
        print('      3D model building completed.')
        print(
            '      3D model building time: ',
            np.round((end_1 - start_1) / 60, 2),
            ' minutes',
        )

        chk_tri = pd.DataFrame(chk_tri, columns=['ID', 'Result', 'SMILES'])
        chk_tri.to_csv(list_out_xyz)

        bd.del_tmp_files()

        # Delete work directory
        if os.path.isdir('work_dir/'):
            shutil.rmtree('work_dir/')

        return chk_tri
