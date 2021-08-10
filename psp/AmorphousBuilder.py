import numpy as np
import pandas as pd
import psp.MD_lib as MDlib
import time
import os
import psp.PSP_lib as bd
from openbabel import openbabel as ob
from subprocess import call
import glob

# from scipy.optimize import minimize
from optimparallel import minimize_parallel
import psp.MoleculeBuilder as mb
import random


class Builder:
    def __init__(
        self,
        Dataframe,
        ID_col="ID",
        SMILES_col="smiles",
        NumMole="Num",
        Length="Len",
        NumConf="NumConf",
        NumModel=1,
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
        self.NumModel = NumModel
        self.LeftCap = LeftCap
        self.RightCap = RightCap
        self.Loop = Loop
        self.OutFile = OutFile
        self.OutDir = os.path.join(OutDir, "")
        self.OutDir_xyz = os.path.join(OutDir, OutDir_xyz, "")
        self.OutDir_packmol = os.path.join(OutDir, "packmol", "")
        self.OutDir_ligpargen = os.path.join(OutDir, "ligpargen", "")
        self.OutDir_pysimm = os.path.join(OutDir, "pysimm", "")
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

        # if self.NumModel > 1:
        #    for model in range(1,self.NumModel+1):
        #
        # else:

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
                NumConf=int(df[self.NumConf].values) * self.NumModel,
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

        XYZ_list, smi_list, NMol_list, NumConf_list = [], [], [], []
        for index, row in self.Dataframe.iterrows():
            # Get number of molecules for each conformer of molecules
            NMol_list += [int(row[self.NumMole] / row[self.NumConf])] * row[
                self.NumConf
            ]

            # Get SMILES string for oligomers
            smiles_each = xyz_gen_pd[xyz_gen_pd['ID'] == row['ID']]['SMILES'].values[0]
            smi_list += smiles_each * row[self.NumConf]

            # Get a list of filenames for XYZ coordinates
            XYZ_list_ind = glob.glob(self.OutDir_xyz + str(row[self.ID_col]) + "*.pdb")
            XYZ_list.append(XYZ_list_ind)
            NumConf_list.append(int(row[self.NumConf]))

            # for conf in range(1, row[self.NumConf] + 1):
            #    XYZ_list.append(
            #        self.OutDir_xyz
            #        + str(row[self.ID_col])
            #        + "_N"
            #        + str(row[self.Length])
            #        + "_C"
            #        + str(conf)
            #        + ".pdb"
            #    )

        # Define boundary conditions
        if max(self.box_size) == 0.0:  # Box size is not provided
            NMol_type = len(NMol_list)
            Total_NMol = sum(NMol_list)
            total_vol = 0
            for i in range(NMol_type):
                molar_mass = MDlib.get_molar_mass(smi_list[i])
                total_vol += MDlib.get_vol(self.density, NMol_list[i], molar_mass)
            self.box_size = MDlib.get_box_size(
                total_vol, box_type=self.box_type, incr_per=self.incr_per
            )

        xmin, xmax, ymin, ymax, zmin, zmax = (
            self.box_size[0],
            self.box_size[1],
            self.box_size[2],
            self.box_size[3],
            self.box_size[4],
            self.box_size[5],
        )

        fix_dis = self.tol_dis / 2

        ind_mol_count = [0] * len(NumConf_list)
        count_model = 0
        for model in range(1, self.NumModel + 1):
            if self.NumModel > 1:
                print("MODEL ", model)
                packmol_outdir_model = self.OutDir_packmol[:-1] + '_' + str(model) + "/"
                bd.build_dir(packmol_outdir_model)

                XYZ_list_ind_model = []
                count_mol = 0
                for ind_list in XYZ_list:
                    if len(ind_list) >= (count_model + 1) * NumConf_list[count_mol]:
                        XYZ_list_ind_model.append(
                            ind_list[
                                count_model
                                * NumConf_list[count_mol]: (count_model + 1)
                                * NumConf_list[count_mol]
                            ]
                        )
                    else:
                        XYZ_list_ind_model.append(
                            random.sample(ind_list, NumConf_list[count_mol])
                        )

                    count_mol += 1

                XYZ_list_model = [
                    item for sublist in XYZ_list_ind_model for item in sublist
                ]
                count_model += 1
            else:
                bd.build_dir(self.OutDir_packmol)

                packmol_outdir_model = self.OutDir_packmol
                XYZ_list_model = [item for sublist in XYZ_list for item in sublist]
            # exit()
            # PACKMOL input file
            MDlib.gen_packmol_inp(
                packmol_outdir_model,
                self.tol_dis,
                XYZ_list_model,
                NMol_list,
                xmin + fix_dis,
                xmax - fix_dis,
                ymin + fix_dis,
                ymax - fix_dis,
                zmin + fix_dis,
                zmax - fix_dis,
            )

            # PACKMOL calculation
            command = (
                packmol_path + " < " + os.path.join(packmol_outdir_model, "packmol.inp")
            )
            errout = MDlib.run_packmol(
                command, os.path.join(packmol_outdir_model, "packmol.out")
            )

            if errout is not None:
                print(" Error in packmol calculation")
                exit()
            elif (
                os.path.exists(os.path.join(packmol_outdir_model, "packmol.pdb"))
                is False
            ):
                print(" Error in packmol calculation")
                exit()

            mol = ob.OBMol()
            obConversion = ob.OBConversion()
            obConversion.SetInAndOutFormats("pdb", "mol2")
            obConversion.ReadFile(
                mol, os.path.join(packmol_outdir_model, "packmol.pdb")
            )
            obConversion.WriteFile(
                mol, os.path.join(packmol_outdir_model, "packmol.mol2")
            )

            packmol_xyz = MDlib.read_mol2_xyz(
                os.path.join(packmol_outdir_model, "packmol.mol2")
            )
            packmol_bond = MDlib.read_mol2_bond(
                os.path.join(packmol_outdir_model, "packmol.mol2")
            )
            # packmol_xyz = pd.read_csv(
            #    self.OutDir_packmol + "packmol.xyz",
            #    header=None,
            #    skiprows=2,
            #    delim_whitespace=True,
            # )

            # Output filename
            if self.NumModel > 1:
                output_filename = self.OutFile + "_N" + str(count_model)
            else:
                output_filename = self.OutFile

            MDlib.gen_sys_vasp(
                os.path.join(self.OutDir, output_filename + ".vasp"),
                packmol_xyz,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
            )
            MDlib.gen_sys_data(
                os.path.join(self.OutDir, output_filename + ".data"),
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

    def get_opls(self, output_fname='amor_opls.lmps'):
        system_pdb_fname = os.path.join(self.OutDir_packmol, "packmol.pdb")
        r = MDlib.get_coord_from_pdb(system_pdb_fname)

        bd.build_dir(self.OutDir_ligpargen)

        system_stats = {
            'total_atoms': 0,
            'total_bonds': 0,
            'total_angles': 0,
            'total_dihedrals': 0,
            'total_impropers': 0,
            'total_atom_types': 0,
            'total_bond_types': 0,
            'total_angle_types': 0,
            'total_dihedral_types': 0,
            'total_improper_types': 0,
        }
        dicts = []

        # run LigParGen for every pdb file in the OutDir_xyz directory
        for index, row in self.Dataframe.iterrows():
            _id = str(row[self.ID_col])
            _length = row[self.Length]
            _num = row[self.NumMole]
            _conf = 1  # read in only the first conformer
            output_prefix = "{}_N{}_C{}".format(_id, _length, _conf)
            lig_output_fname = "{}.lmp".format(output_prefix)
            data_fname = os.path.join(self.OutDir_ligpargen, lig_output_fname)

            try:
                print("LigParGen working on {}.pdb".format(output_prefix))
                call(
                    "LigParGen -p {0}{1}.pdb -r {1} -c 0 -o 0 -l".format(
                        self.OutDir_xyz, output_prefix
                    ),
                    shell=True,
                )
                os.rename(lig_output_fname, data_fname)
            except BaseException:
                print('problem running LigParGen for {}.pdb.'.format(output_prefix))

            # quickly read the headers of LigParGen generated LAMMPS
            # files to count total number of atoms/bonds/angles...etc
            (
                natoms,
                nbonds,
                nangles,
                ndihedrals,
                nimpropers,
                natom_types,
                nbond_types,
                nangle_types,
                ndihedral_types,
                nimproper_types,
            ) = MDlib.read_lmps_header(data_fname)

            system_stats['total_atom_types'] += natom_types
            system_stats['total_bond_types'] += nbond_types
            system_stats['total_angle_types'] += nangle_types
            system_stats['total_dihedral_types'] += ndihedral_types
            system_stats['total_improper_types'] += nimproper_types
            system_stats['total_atoms'] += natoms * _num
            system_stats['total_bonds'] += nbonds * _num
            system_stats['total_angles'] += nangles * _num
            system_stats['total_dihedrals'] += ndihedrals * _num
            system_stats['total_impropers'] += nimpropers * _num

            # this switcher dict is to navigate through and store info for each section of a LAMMPS file
            switcher = {
                'Masses': [],
                'Pair Coeffs': [],
                'Bond Coeffs': [],
                'Angle Coeffs': [],
                'Dihedral Coeffs': [],
                'Improper Coeffs': [],
                'Atoms': [],
                'Bonds': [],
                'Angles': [],
                'Dihedrals': [],
                'Impropers': [],
                'Num': _num,
            }
            current_section = None

            # read all the info in the LigParGen generated LAMMPS file for modification
            with open(data_fname, 'rt') as lines:
                for line in lines:
                    if any(x in line for x in switcher.keys()):
                        current_section = line.strip()
                    elif line == '\n' or not current_section:
                        continue
                    else:
                        section_list = switcher.get(
                            current_section, 'Invalid current section'
                        )
                        section_list.append(line.split())
            dicts.append(switcher)

        lammps_output = os.path.join(self.OutDir, output_fname)
        MDlib.write_lammps_ouput(lammps_output, r, self.box_size, system_stats, dicts)

    def get_gaff2(
        self, output_fname='amor_gaff2.lmps', atom_typing='pysimm', swap_dict=None
    ):
        system_pdb_fname = os.path.join(self.OutDir_packmol, "packmol.pdb")
        r = MDlib.get_coord_from_pdb(system_pdb_fname)

        bd.build_dir(self.OutDir_pysimm)

        system_stats = {
            'total_atoms': 0,
            'total_bonds': 0,
            'total_angles': 0,
            'total_dihedrals': 0,
            'total_impropers': 0,
            'total_atom_types': 0,
            'total_bond_types': 0,
            'total_angle_types': 0,
            'total_dihedral_types': 0,
            'total_improper_types': 0,
        }
        dicts = []

        from pysimm import system, forcefield

        # run Pysimm for every mol2 (converted from pdb with Babel) file in the OutDir_xyz directory
        for index, row in self.Dataframe.iterrows():
            _id = str(row[self.ID_col])
            _length = row[self.Length]
            _num = row[self.NumMole]
            _conf = 1  # read in only the first conformer
            output_prefix = "{}_N{}_C{}".format(_id, _length, _conf)
            mol2_file = os.path.join(self.OutDir_xyz, "{}.mol2".format(output_prefix))
            call(
                'babel -ipdb {0}.pdb -omol2 {0}.mol2'.format(
                    os.path.join(self.OutDir_xyz, output_prefix)
                ),
                shell=True,
            )
            data_fname = os.path.join(
                self.OutDir_pysimm, "{}.lmp".format(output_prefix)
            )

            try:
                print("Pysimm working on {}".format(mol2_file))
                s = system.read_mol2(mol2_file)
            except BaseException:
                print('problem reading {} for Pysimm.'.format(mol2_file))

            f = forcefield.Gaff2()
            if atom_typing == 'pysimm':
                try:
                    print("Pysimm applying force field for {}.".format(mol2_file))
                    s.apply_forcefield(f, charges='gasteiger')
                except BaseException:
                    print(
                        'Error applying force field with the mol2 file, switch to using cml file.'
                    )
                    call(
                        'babel -ipdb {0}.pdb -ocml {0}.cml'.format(
                            os.path.join(self.OutDir_xyz, output_prefix)
                        ),
                        shell=True,
                    )
                    s = system.read_cml(
                        '{}.cml'.format(os.path.join(self.OutDir_xyz, output_prefix))
                    )
                    for b in s.bonds:
                        if b.a.bonds.count == 3 and b.b.bonds.count == 3:
                            b.order = 4
                    s.apply_forcefield(f, charges='gasteiger')
            elif atom_typing == 'antechamber':
                MDlib.get_type_from_antechamber(s, mol2_file, 'gaff2', f, swap_dict)
                s.pair_style = 'lj'
                s.apply_forcefield(f, charges='gasteiger', skip_ptypes=True)
            else:
                print(
                    'Invalid atom typing option, please select pysimm or antechamber.'
                )
            s.write_lammps(data_fname)

            # quickly read the headers of Pysimm generated LAMMPS
            # files to count total number of atoms/bonds/angles...etc
            (
                natoms,
                nbonds,
                nangles,
                ndihedrals,
                nimpropers,
                natom_types,
                nbond_types,
                nangle_types,
                ndihedral_types,
                nimproper_types,
            ) = MDlib.read_lmps_header(data_fname)

            system_stats['total_atom_types'] += natom_types
            system_stats['total_bond_types'] += nbond_types
            system_stats['total_angle_types'] += nangle_types
            system_stats['total_dihedral_types'] += ndihedral_types
            system_stats['total_improper_types'] += nimproper_types
            system_stats['total_atoms'] += natoms * _num
            system_stats['total_bonds'] += nbonds * _num
            system_stats['total_angles'] += nangles * _num
            system_stats['total_dihedrals'] += ndihedrals * _num
            system_stats['total_impropers'] += nimpropers * _num

            # this switcher dict is to navigate through and store info for each section of a LAMMPS file
            switcher = {
                'Masses': [],
                'Pair Coeffs': [],
                'Bond Coeffs': [],
                'Angle Coeffs': [],
                'Dihedral Coeffs': [],
                'Improper Coeffs': [],
                'Atoms': [],
                'Velocities': [],
                'Bonds': [],
                'Angles': [],
                'Dihedrals': [],
                'Impropers': [],
                'Num': _num,
            }
            current_section = None

            # read all the info in the Pysimm generated LAMMPS file for modification
            with open(data_fname, 'rt') as lines:
                for line in lines:
                    if any(x in line for x in switcher.keys()):
                        current_section = line.strip()
                    elif line == '\n' or not current_section:
                        continue
                    else:
                        section_list = switcher.get(
                            current_section, 'Invalid current section'
                        )
                        section_list.append(line.split())
            dicts.append(switcher)

        lammps_output = os.path.join(self.OutDir, output_fname)
        MDlib.write_lammps_ouput(lammps_output, r, self.box_size, system_stats, dicts)
