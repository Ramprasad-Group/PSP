import numpy as np
import pandas as pd
import MD_lib as MDlib
import time
import os
import PSP_lib as bd
from openbabel import openbabel as ob
from subprocess import call

# from scipy.optimize import minimize
from optimparallel import minimize_parallel
import psp.MoleculeBuilder as mb


class Builder:
    def __init__(
        self,
        Dataframe,
        ID_col="ID",
        SMILES_col="smiles",
        NumMole="Num",
        Length="Len",
        NumConf="NumConf",
        LeftCap="LeftCap",
        RightCap="RightCap",
        Loop="Loop",
        OutFile="amor_model",
        OutDir="amorphous_models",
        OutDir_xyz="molecules",
        density=0.65,
        tol_dis=2.0,
        box_type="c",
        box_size=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        incr_per=0.4,
        BondInfo=True,
    ):
        self.Dataframe = Dataframe
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.NumMole = NumMole
        self.Length = Length
        self.NumConf = NumConf
        self.LeftCap = (LeftCap,)
        self.RightCap = (RightCap,)
        self.Loop = Loop
        self.OutFile = OutFile
        self.OutDir = OutDir + "/"
        self.OutDir_xyz = OutDir + "/" + OutDir_xyz + "/"
        self.OutDir_packmol = OutDir + "/" + "packmol" + "/"
        self.OutDir_ligpargen = OutDir + "/" + "ligpargen" + "/"
        self.density = density
        self.tol_dis = tol_dis
        self.box_type = box_type
        self.box_size = box_size
        self.incr_per = incr_per
        self.BondInfo = BondInfo
    def Build(self):
        # location of directory for VASP inputs (polymers) and build a directory
        bd.build_dir(self.OutDir)
        bd.build_dir(self.OutDir_xyz)
        bd.build_dir(self.OutDir_packmol)

        # PACKMOL
        packmol_path = os.getenv("PACKMOL_EXEC")
        # packmol_path = '/home/hari/.soft/packmol/packmol'

        xyz_gen_pd = pd.DataFrame()
        for i in self.Dataframe.index:
            df = pd.DataFrame(self.Dataframe.loc[i]).T

            mol = mb.Builder(
                df,
                ID_col=self.ID_col,
                SMILES_col=self.SMILES_col,
                LeftCap=self.LeftCap[0],
                RightCap=self.RightCap[0],
                OutDir=self.OutDir_xyz,
                Length=[int(df[self.Length].values)],
                NumConf=int(df[self.NumConf].values),
                Loop=eval(str(df[self.Loop].values[0])),
                NCores=-1,
            )
            results = mol.Build()
            xyz_gen_pd = pd.concat([xyz_gen_pd, results])

        if len(list(set(xyz_gen_pd["Result"].values))) != 1:
            xyz_gen_pd.to_csv("molecules.csv")
            print(
                "Couldn't generate XYZ coordinates of molecules, check 'molecules.csv'"
            )

        XYZ_list, smi_list, NMol_list = [], [], []
        for index, row in self.Dataframe.iterrows():
            # Get number of molecules for each conformer of molecules
            NMol_list += [int(row[self.NumMole] / row[self.NumConf])] * row[
                self.NumConf
            ]

            # Get SMILES string for oligomers
            smiles_each = xyz_gen_pd[xyz_gen_pd['ID'] == row['ID']]['SMILES'].values[0]
            smi_list += smiles_each * row[self.NumConf]

            # Get a list of filenames for XYZ coordinates
            for conf in range(1, row[self.NumConf] + 1):
                XYZ_list.append(
                    self.OutDir_xyz
                    + str(row[self.ID_col])
                    + "_N"
                    + str(row[self.Length])
                    + "_C"
                    + str(conf)
                    + ".pdb"
                )

        # Define boundary conditions
        if max(self.box_size) == 0.0:  # Box size is not provided
            NMol_type = len(NMol_list)
            Total_NMol = sum(NMol_list)
            total_vol = 0
            for i in range(NMol_type):
                molar_mass = MDlib.get_molar_mass(smi_list[i])
                total_vol += MDlib.get_vol(self.density, NMol_list[i], molar_mass)
            xmin, xmax, ymin, ymax, zmin, zmax = MDlib.get_box_size(
                total_vol, box_type=self.box_type, incr_per=self.incr_per
            )
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = (
                self.box_size[0],
                self.box_size[1],
                self.box_size[2],
                self.box_size[3],
                self.box_size[4],
                self.box_size[5],
            )

        fix_dis = self.tol_dis / 2

        # PACKMOL input file
        MDlib.gen_packmol_inp(
            self.OutDir_packmol,
            self.tol_dis,
            XYZ_list,
            NMol_list,
            xmin + fix_dis,
            xmax - fix_dis,
            ymin + fix_dis,
            ymax - fix_dis,
            zmin + fix_dis,
            zmax - fix_dis,
        )

        # PACKMOL calculation
        command = packmol_path + " < " + self.OutDir_packmol + "packmol.inp"
        errout = MDlib.run_packmol(command, self.OutDir_packmol + "packmol.out")

        if errout is not None:
            print(" Error in packmol calculation")
            exit()
        elif os.path.exists(self.OutDir_packmol + "packmol.pdb") is False:
            print(" Error in packmol calculation")
            exit()

        mol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "mol2")
        obConversion.ReadFile(mol, self.OutDir_packmol + "packmol.pdb")
        obConversion.WriteFile(mol, self.OutDir_packmol + "packmol.mol2")

        packmol_xyz = MDlib.read_mol2_xyz(self.OutDir_packmol + "packmol.mol2")
        packmol_bond = MDlib.read_mol2_bond(self.OutDir_packmol + "packmol.mol2")
        # packmol_xyz = pd.read_csv(
        #    self.OutDir_packmol + "packmol.xyz",
        #    header=None,
        #    skiprows=2,
        #    delim_whitespace=True,
        # )

        MDlib.gen_sys_vasp(
            self.OutDir + self.OutFile + ".vasp",
            packmol_xyz,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        )
        MDlib.gen_sys_data(
            self.OutDir + self.OutFile + ".data",
            packmol_xyz,
            packmol_bond,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            self.BondInfo,
        )

    def Build_psp(self):
        # location of directory for VASP inputs (polymers) and build a directory
        bd.build_dir(self.OutDir)
        bd.build_dir(self.OutDir_xyz)

        xyz_gen_pd = pd.DataFrame()
        for i in self.Dataframe.index:
            df = pd.DataFrame(self.Dataframe.loc[i]).T

            mol = mb.Builder(
                df,
                ID_col=self.ID_col,
                SMILES_col=self.SMILES_col,
                OutDir=self.OutDir_xyz,
                Length=[int(df[self.Length].values)],
                NumConf=int(df[self.NumConf].values),
                Loop=eval(str(df[self.Loop].values[0])),
            )
            results = mol.Build3D()
            xyz_gen_pd = pd.concat([xyz_gen_pd, results])

        if len(list(set(xyz_gen_pd["Result"].values))) != 1:
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
                    row[self.ID_col], row[self.SMILES_col], self.OutDir_xyz, self.Length
                )
                smiles_each = bd.gen_oligomer_smiles(
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    row[self.SMILES_col],
                    row[self.Length],
                    eval(str(row[self.Loop])),
                )
            else:
                smiles_each = row[self.SMILES_col]
            smi_list += [smiles_each] * row[self.NumConf]

            # Get a list of filenames for XYZ coordinates
            for conf in range(1, row[self.NumConf] + 1):
                XYZ_list.append(
                    self.OutDir_xyz
                    + "/"
                    + str(row[self.ID_col])
                    + "_N"
                    + str(row[self.Length])
                    + "_C"
                    + str(conf)
                    + ".xyz"
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
        disx = list(np.random.uniform(0, 4, Total_NMol))  # [0] * Total_NMol
        disy = list(np.random.uniform(0, 4, Total_NMol))  # [0] * Total_NMol
        disz = list(np.random.uniform(0, 4, Total_NMol))  # [0] * Total_NMol
        theta1 = list(np.random.uniform(0, 180, Total_NMol))
        theta2 = list(np.random.uniform(0, 180, Total_NMol))
        theta3 = list(np.random.uniform(0, 180, Total_NMol))
        x0 = disx + disy + disz + theta1 + theta2 + theta3

        sys = MDlib.get_initial_model(
            NMol_list, XYZ_list, self.tol_dis, xmin, xmax, ymin, ymax, zmin, zmax
        )
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
                options={"disp": True},
            )
            end_1 = time.time()
            print(
                " minimize time: ", np.round((end_1 - start_1) / 60, 2), " minutes",
            )

            solution = res["x"]
            evaluation = MDlib.main_func(
                solution, sys, self.tol_dis, xmin, xmax, ymin, ymax, zmin, zmax
            )
            print(evaluation)
            arr_x = np.array_split(solution, 6)
            sys = MDlib.move_molecules(
                sys, arr_x[0], arr_x[1], arr_x[2], arr_x[3], arr_x[4], arr_x[5]
            )
            MDlib.gen_sys_xyz(self.OutDir + self.OutFile + ".xyz", sys)
            MDlib.gen_sys_vasp(
                self.OutDir + self.OutFile + ".vasp",
                sys,
                xmin - proxy_dis,
                xmax + proxy_dis,
                ymin - proxy_dis,
                ymax + proxy_dis,
                zmin - proxy_dis,
                zmax + proxy_dis,
            )
            MDlib.gen_sys_data(
                self.OutDir + self.OutFile + ".data",
                sys,
                xmin - proxy_dis,
                xmax + proxy_dis,
                ymin - proxy_dis,
                ymax + proxy_dis,
                zmin - proxy_dis,
                zmax + proxy_dis,
            )
        else:
            print("Value of the Objective function: ", evaluation)
        sys1 = MDlib.move_molecules(sys, disx, disy, disz, theta1, theta2, theta3)
        MDlib.gen_sys_xyz(self.OutDir + "initial_geo.xyz", sys1)
        MDlib.gen_sys_vasp(
            self.OutDir + "initial_geo.vasp",
            sys1,
            xmin - proxy_dis,
            xmax + proxy_dis,
            ymin - proxy_dis,
            ymax + proxy_dis,
            zmin - proxy_dis,
            zmax + proxy_dis,
        )

    def get_opls_param(self):
        bd.build_dir(self.OutDir_ligpargen)

        # run LigParGen for every pdb file in the OutDir_xyz directory    
        for index, row in self.Dataframe.iterrows():
            for conf in range(1, row[self.NumConf] + 1):
                _id = str(row[self.ID_col])
                _length = row[self.Length]
                _conf = str(conf)
                conf_pdb_fname = self.OutDir_xyz + "{}_N{}_C{}.pdb".format(_id, _length, _conf)

                try:
                    print("LigParGen working on {}". format(conf_pdb_fname))
                    call("LigParGen -p {} -r {} -c 0 -o 0 -l".format(conf_pdb_fname, _id), shell=True)
                    lig_output_fname = "{}.lmp".format(_id)
                    os.rename(lig_output_fname, "{}{}".format(self.OutDir_ligpargen, lig_output_fname))
                except BaseException:
                    print('problem running LigParGen for {}.'.format(conf_pdb_fname))

        system_pdb_fname = self.OutDir_packmol + "packmol.pdb"
        skip_beginning = 5 # header lines of packmol.pdb
        atom_count = 0 # coutner for atom number
        r = np.zeros([1, 3], float) # 2D array of x, y, z coordinates, r[id][coordinate]

        # get all atom coordinates from the system/packmol pdb file
        with open(system_pdb_fname, 'r') as f:
            for skipped_frame in range(skip_beginning):
                f.readline()

            line = f.readline()
            x_coord, y_coord, z_coord = MDlib.read_pdb_line(line)
            r[atom_count][0] = x_coord
            r[atom_count][1] = y_coord
            r[atom_count][2] = z_coord

            # if next line still returns x, y, z coordinates, allocate more memeory for the array
            while True:
                try:
                    atom_count += 1
                    line = f.readline()
                    x_coord, y_coord, z_coord = MDlib.read_pdb_line(line)
                    r = np.concatenate( ( r, np.zeros([1, 3], float) ) )
                    r[atom_count][0] = x_coord
                    r[atom_count][1] = y_coord
                    r[atom_count][2] = z_coord
                except:
                    break
