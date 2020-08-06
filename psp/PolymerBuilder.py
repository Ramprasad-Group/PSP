import numpy as np
import pandas as pd
import psp.PB_lib as bd
import openbabel as ob
import os
import shutil
import time
import multiprocessing
from joblib import Parallel, delayed
obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("mol", "xyz")

class Builder:
    def __init__(self, df_smiles, num_conf=1, length=['n'], input_monomer_angles='medium', input_dimer_angles='low', Steps=20, Substeps=10, n_cores=0, method='SA', ID_col='ID', SMILES_col='smiles', OutDir='chains'):
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.OutDir = OutDir
        self.df_smiles = df_smiles
        self.num_conf = num_conf
        self.length = length
        self.input_monomer_angles = input_monomer_angles
        self.input_dimer_angles = input_dimer_angles
        self.Steps = Steps
        self.Substeps = Substeps
        self.n_cores = n_cores
        self.method = method
        if self.method in ['SA', 'Dimer']:
#            print('oooooo')
            print('     polymer chain building started (', self.method,') ...')
        else:
            print("Error: please check keyword for method")
            print("SA == simulated annealing")
            print("SMART == Constraint optimization")
            print("Dimer == dimerization")
            exit()
# list of molecules name and CORRECT/WRONG
    def BuildPolymer(self):
        # Input Parameters
        intense = np.arange(-180,180,10)
        medium = [30, -30, 45, -45, 60, -60, 90, 120, -120, 135, -135, 150, -150, 180]
        low = [45, -45, 60, -60, 90, 120, -120, 180]
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
        chk_tri=[]
        ID=self.ID_col
        SMILES=self.SMILES_col
        df=self.df_smiles.copy()
        df[ID] = df[ID].apply(str)

        rot_angles_monomer = vars()[self.input_monomer_angles]
        rot_angles_dimer = vars()[self.input_dimer_angles]

        if self.n_cores == 0:
            self.n_cores = multiprocessing.cpu_count() - 1

        result=Parallel(n_jobs=self.n_cores)(delayed(bd.build_polymer)(unit_name, df, ID, SMILES, xyz_in_dir, xyz_tmp_dir,
            vasp_out_dir, rot_angles_monomer, rot_angles_dimer, self.Steps, self.Substeps, self.num_conf, self.length, self.method)
            for unit_name in df[ID].values)
        for i in result:
            chk_tri.append([i[0],i[1],i[2]])
        end_1 = time.time()
        print("")
        print('      polymer chain building completed.')
        print('      polymer chain building time: ', np.round((end_1-start_1)/60,2), ' minutes')

        chk_tri = pd.DataFrame(chk_tri, columns=['ID', 'Result','Conformers'])
        chk_tri.to_csv(list_out_xyz)

        ### Delete work directory
        if os.path.isdir('work_dir/'):
            shutil.rmtree('work_dir/')
        return chk_tri
