import numpy as np
import pandas as pd
import psp.PSP_lib as bd
from openbabel import openbabel as ob
import os
import shutil
import time
import multiprocessing
from joblib import Parallel, delayed

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("mol", "xyz")


class Builder:
    def __init__(
        self,
        Dataframe,
        NumConf=1,
        Length=['n'],
        MonomerAng='medium',
        DimerAng='low',
        Steps=20,
        Substeps=10,
        NCores=0,
        Method='SA',
        ID_col='ID',
        SMILES_col='smiles',
        IntraChainCorr=1,
        Tol_ChainCorr=50,
        OutDir='chains',
    ):
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.OutDir = OutDir
        self.Dataframe = Dataframe
        self.NumConf = NumConf
        self.Length = Length
        self.MonomerAng = MonomerAng
        self.DimerAng = DimerAng
        self.Steps = Steps
        self.Substeps = Substeps
        self.NCores = NCores
        self.Method = Method
        self.IntraChainCorr = IntraChainCorr
        self.Tol_ChainCorr = Tol_ChainCorr

        if self.Method in ['SA', 'Dimer']:
            print('    - polymer chain building started (', self.Method, ') ...')
        else:
            print("Error: please check keyword for method")
            print("SA == simulated annealing")
            print("SMART == Constraint optimization")
            print("Dimer == dimerization")
            exit()

    # list of molecules name and CORRECT/WRONG
    def BuildChain(self):
        # Input Parameters
        intense = np.arange(-180, 180, 10)
        medium = [
            0,
            30,
            -30,
            45,
            -45,
            60,
            -60,
            90,
            120,
            -120,
            135,
            -135,
            150,
            -150,
            180,
        ]
        low = [0, 45, -45, 60, -60, 90, 120, -120, 180]

        # Directories
        # Working directory
        bd.build_dir('work_dir/')

        # location of input XYZ files
        xyz_in_dir = 'work_dir/xyz-in/'

        xyz_tmp_dir = 'work_dir/xyz-temp/'
        bd.build_dir(xyz_in_dir)
        bd.build_dir(xyz_tmp_dir)

        # location of directory for VASP inputs (polymers) and build a directory
        vasp_out_dir = self.OutDir + '/'
        bd.build_dir(vasp_out_dir)

        start_1 = time.time()
        list_out_xyz = 'output.csv'
        chk_tri = []
        ID = self.ID_col
        SMILES = self.SMILES_col
        df = self.Dataframe.copy()
        df[ID] = df[ID].apply(str)

        rot_angles_monomer = vars()[self.MonomerAng]
        rot_angles_dimer = vars()[self.DimerAng]

        if self.NCores == 0:
            self.NCores = multiprocessing.cpu_count() - 1

        result = Parallel(n_jobs=self.NCores)(
            delayed(bd.build_polymer)(
                unit_name,
                df,
                ID,
                SMILES,
                xyz_in_dir,
                xyz_tmp_dir,
                vasp_out_dir,
                rot_angles_monomer,
                rot_angles_dimer,
                self.Steps,
                self.Substeps,
                self.NumConf,
                self.Length,
                self.Method,
                self.IntraChainCorr,
                self.Tol_ChainCorr,
            )
            for unit_name in df[ID].values
        )
        for i in result:
            chk_tri.append([i[0], i[1], i[2]])

        end_1 = time.time()
        print("")
        print('    - polymer chain building completed.')
        print(
            '    - polymer chain building time: ',
            np.round((end_1 - start_1) / 60, 2),
            ' minutes',
        )

        chk_tri = pd.DataFrame(chk_tri, columns=['ID', 'Result', 'Conformers'])
        chk_tri.to_csv(list_out_xyz)

        # Delete work directory
        if os.path.isdir('work_dir/'):
            shutil.rmtree('work_dir/')
        return chk_tri
