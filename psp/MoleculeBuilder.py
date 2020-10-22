import numpy as np
import pandas as pd
import psp.PSP_lib as bd
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
        OutDir='molecule',
    ):
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.OutDir = OutDir
        self.Dataframe = Dataframe
        self.NCores = NCores

    # list of molecules name and CORRECT/WRONG
    def Build3D(self):

        # location of directory for VASP inputs (polymers) and build a directory
        out_dir = self.OutDir + '/'
        bd.build_dir(out_dir)

        start_1 = time.time()
        list_out_xyz = 'output.csv'
        chk_tri = []
        ID = self.ID_col
        SMILES = self.SMILES_col
        df = self.Dataframe.copy()
        df[ID] = df[ID].apply(str)

        if self.NCores == 0:
            self.NCores = multiprocessing.cpu_count() - 1

        result = Parallel(n_jobs=self.NCores)(
            delayed(bd.build_3D)(unit_name, df, ID, SMILES, out_dir)
            for unit_name in df[ID].values
        )
        for i in result:
            chk_tri.append([i[0], i[1]])

        end_1 = time.time()
        print("")
        print('      3D model building completed.')
        print(
            '      3D model building time: ',
            np.round((end_1 - start_1) / 60, 2),
            ' minutes',
        )

        chk_tri = pd.DataFrame(chk_tri, columns=['ID', 'Result'])
        chk_tri.to_csv(list_out_xyz)

        return chk_tri
