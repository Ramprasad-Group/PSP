import numpy as np
import pandas as pd
import psp.MD_lib as MDlib
import time
import psp.PSP_lib as bd
from optimparallel import minimize_parallel
import psp.MoleculeBuilder as mb


class Builder:
    def __init__(
        self,
        Dataframe,
        ID_col='ID',
        SMILES_col='smiles',
        NumMole='Num',
        Length='Len',
        NumConf='NumConf',
        loop=False,
        OutFile='amor_model',
        OutDir='amorphous_models',
        OutDir_xyz='molecules',
        density=0.65,
        tol_dis=2.0,
        box_type='c',
        incr_per=0.4,
    ):
        self.Dataframe = Dataframe
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.NumMole = NumMole
        self.Length = Length
        self.NumConf = NumConf
        self.loop = loop
        self.OutFile = OutFile
        self.OutDir = OutDir
        self.OutDir_xyz = OutDir_xyz
        self.density = density
        self.tol_dis = tol_dis
        self.box_type = box_type
        self.incr_per = incr_per

    def Build(self):
        # location of directory for VASP inputs (polymers) and build a directory
        out_dir = self.OutDir + '/'
        bd.build_dir(out_dir)
        OutDir_xyz = out_dir + self.OutDir_xyz + '/'
        bd.build_dir(OutDir_xyz)

        xyz_gen_pd = pd.DataFrame()
        for i in self.Dataframe.index:
            df = pd.DataFrame(self.Dataframe.loc[i]).T
            mol = mb.Builder(
                df,
                ID_col=self.ID_col,
                SMILES_col=self.SMILES_col,
                OutDir=OutDir_xyz,
                Length=[int(df[self.Length].values)],
                NumConf=int(df[self.NumConf].values),
                loop=self.loop,
            )
            results = mol.Build3D()
            xyz_gen_pd = pd.concat([xyz_gen_pd, results])

        if len(list(set(xyz_gen_pd['Result'].values))) != 1:
            xyz_gen_pd.to_csv("molecules.csv")
            print(
                "Couldn't generate XYZ coordinates of molecules, check 'molecules.csv'"
            )
            exit()

        XYZ_list, smi_list, NMol_list = [], [], []
        for index, row in self.Dataframe.iterrows():
            # Get number of molecules for each conformer of molecules
            NMol_list += [int(row[self.NumMole] / row[self.NumConf])] * row[
                self.NumConf
            ]

            # Get SMILES string for oligomers
            # Get row number for dummy and linking atoms
            if row[self.Length] > 1:
                (
                    unit_name,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    m1,
                    neigh_atoms_info,
                    oligo_list,
                    dum,
                    unit_dis,
                    flag,
                ) = bd.Init_info(
                    row[self.ID_col], row[self.SMILES_col], OutDir_xyz, self.Length
                )
                smiles_each = bd.gen_oligomer_smiles(
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    row[self.SMILES_col],
                    row[self.Length],
                    self.loop,
                )
            else:
                smiles_each = row[self.SMILES_col]
            smi_list += [smiles_each] * row[self.NumConf]

            # Get a list of filenames for XYZ coordinates
            for conf in range(1, row[self.NumConf] + 1):
                XYZ_list.append(
                    OutDir_xyz
                    + '/'
                    + str(row[self.ID_col])
                    + '_N'
                    + str(row[self.Length])
                    + '_C'
                    + str(conf)
                    + '.xyz'
                )

        # Define boundary conditions
        NMol_type = len(NMol_list)
        Total_NMol = sum(NMol_list)
        total_vol = 0
        for i in range(NMol_type):
            molar_mass = MDlib.get_molar_mass(smi_list[i])
            total_vol += MDlib.get_vol(self.density, NMol_list[i], molar_mass)
        xmin, xmax, ymin, ymax, zmin, zmax = MDlib.get_box_size(
            total_vol, box_type=self.box_type, incr_per=self.incr_per
        )

        # proxy to handle atom connectivity near boundary
        proxy_dis = self.tol_dis / 2

        # Conditions for initial system
        disx = [0] * Total_NMol
        disy = [0] * Total_NMol
        disz = [0] * Total_NMol
        theta1 = [0] * Total_NMol  # list(np.random.uniform(0, 10, Total_NMol))
        theta2 = [0] * Total_NMol  # list(np.random.uniform(0, 10, Total_NMol))
        theta3 = [0] * Total_NMol  # list(np.random.uniform(0, 10, Total_NMol))
        x0 = disx + disy + disz + theta1 + theta2 + theta3

        sys = MDlib.get_initial_model(
            NMol_list, XYZ_list, self.tol_dis, xmin, xmax, ymin, ymax, zmin, zmax
        )
        # exit()
        # if the value of the objective function > 0.0, then perform optimization
        evaluation = MDlib.main_func(
            x0, sys, self.tol_dis, xmin, xmax, ymin, ymax, zmin, zmax
        )
        if evaluation > 0:
            start_1 = time.time()
            res = minimize_parallel(
                MDlib.main_func,
                x0,
                args=(sys, self.tol_dis, xmin, xmax, ymin, ymax, zmin, zmax),
                options={'disp': True},
            )
            end_1 = time.time()
            print(
                ' minimize time: ', np.round((end_1 - start_1) / 60, 2), ' minutes',
            )

            solution = res['x']
            evaluation = MDlib.main_func(
                solution, sys, self.tol_dis, xmin, xmax, ymin, ymax, zmin, zmax
            )
            print(evaluation)
            arr_x = np.array_split(solution, 6)
            sys = MDlib.move_molecules(
                sys, arr_x[0], arr_x[1], arr_x[2], arr_x[3], arr_x[4], arr_x[5]
            )
            MDlib.gen_sys_xyz(self.OutDir + '/' + self.OutFile + ".xyz", sys)
            MDlib.gen_sys_vasp(
                self.OutDir + '/' + self.OutFile + ".vasp",
                sys,
                xmin - proxy_dis,
                xmax + proxy_dis,
                ymin - proxy_dis,
                ymax + proxy_dis,
                zmin - proxy_dis,
                zmax + proxy_dis,
            )
        else:
            print('Value of the Objective function: ', evaluation)
        sys1 = MDlib.move_molecules(sys, disx, disy, disz, theta1, theta2, theta3)
        MDlib.gen_sys_xyz(self.OutDir + '/' + "initial_geo.xyz", sys1)
        MDlib.gen_sys_vasp(
            self.OutDir + '/' + "initial_geo.vasp",
            sys1,
            xmin - proxy_dis,
            xmax + proxy_dis,
            ymin - proxy_dis,
            ymax + proxy_dis,
            zmin - proxy_dis,
            zmax + proxy_dis,
        )
