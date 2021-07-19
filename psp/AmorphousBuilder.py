import numpy as np
import pandas as pd
import psp.MD_lib as MDlib
import time
import os
import psp.PSP_lib as bd
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


    def get_opls(self, output_fname='amor_opls.lmps'):
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

        bd.build_dir(self.OutDir_ligpargen)
        total_atoms, total_bonds, total_angles, total_dihedrals, total_impropers = (0 for i in range(5))
        total_atom_types, total_bond_types, total_angle_types, total_dihedral_types, total_improper_types = (0 for i in range(5))
        dicts = []

        # run LigParGen for every pdb file in the OutDir_xyz directory    
        for index, row in self.Dataframe.iterrows():
            _id = str(row[self.ID_col])
            _length = row[self.Length]
            _num = row[self.NumMole]
            _conf = 1 # read in only the first conformer
            output_prefix = "{}_N{}_C{}".format(_id, _length, _conf)

            try:
                print("LigParGen working on {}.pdb". format(output_prefix))
                call("LigParGen -p {0}{1}.pdb -r {1} -c 0 -o 0 -l".format(self.OutDir_xyz, output_prefix), shell=True)
                lig_output_fname = "{}.lmp".format(output_prefix)
                os.rename(lig_output_fname, self.OutDir_ligpargen + lig_output_fname)
            except BaseException:
                print('problem running LigParGen for {}.pdb.'.format(output_file_prefix))

            # quickly read the headers of LigParGen generated LAMMPS files to count total number of atoms/bonds/angles...etc
            data_fname = self.OutDir_ligpargen + lig_output_fname
            natoms, nbonds, nangles, ndihedrals, nimpropers = MDlib.read_lmps_header(data_fname)

            total_atom_types += natoms
            total_bond_types += nbonds
            total_angle_types += nangles
            total_dihedral_types += ndihedrals
            total_improper_types += nimpropers
            total_atoms += natoms * _num
            total_bonds += nbonds * _num
            total_angles += nangles * _num
            total_dihedrals += ndihedrals * _num
            total_impropers += nimpropers * _num

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
                'Num': _num
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
                        section_list = switcher.get(current_section, 'Invalid current section')
                        section_list.append(line.split())
            dicts.append(switcher)
        
        # These switcher dicts are for each section of the LAMMPS file that we will build
        atomconvertdicts, bondconvertdicts, angleconvertdicts, dihedralconvertdicts, improperconvertdicts = ([] for i in range(5))
        switcher_coeffs = {
            'Pair Coeffs': [total_atoms, atomconvertdicts],
            'Bond Coeffs': [total_bonds, bondconvertdicts],
            'Angle Coeffs': [total_angles, angleconvertdicts],
            'Dihedral Coeffs': [total_dihedrals, dihedralconvertdicts],
            'Improper Coeffs': [total_impropers, improperconvertdicts]
        }
        switcher_main = {
            'Bonds': [total_bonds, bondconvertdicts],
            'Angles': [total_angles, angleconvertdicts],
            'Dihedrals': [total_dihedrals, dihedralconvertdicts],
            'Impropers': [total_impropers, improperconvertdicts]
        }

        # build the final LAMMPS output
        with open(output_fname, 'wt') as out:
            # header section
            out.write('LAMMPS data file Created by PSP - with LigParGen OPLS parameters\n')
            out.write('\n')
            out.write('{:>12}  atoms\n'.format(total_atoms))
            out.write('{:>12}  bonds\n'.format(total_bonds))
            out.write('{:>12}  angles\n'.format(total_angles))
            out.write('{:>12}  dihedrals\n'.format(total_dihedrals))
            out.write('{:>12}  impropers\n'.format(total_impropers))
            out.write('\n')
            out.write('{:>12}  atom types\n'.format(total_atom_types))
            out.write('{:>12}  bond types\n'.format(total_bond_types))
            out.write('{:>12}  angle types\n'.format(total_angle_types))
            out.write('{:>12}  dihedral types\n'.format(total_dihedral_types))
            out.write('{:>12}  improper types\n'.format(total_improper_types))
            out.write('\n')
            out.write('{:>12}  {:>12} xlo xhi\n'.format(self.box_size[0], self.box_size[1]))
            out.write('{:>12}  {:>12} ylo yhi\n'.format(self.box_size[2], self.box_size[3]))
            out.write('{:>12}  {:>12} zlo zhi\n'.format(self.box_size[4], self.box_size[5]))
            out.write('\n')

            # Masses section
            out.write('Masses\n')
            out.write('\n')
            counter = 0
            for dic in dicts:
                for fields in dic.get('Masses'):
                    counter += 1
                    out.write('{count:>8} {1:>10}\n'.format(*fields, count=counter))
            out.write('\n')

            # Pair, Bond, Angle, Dihedral, and Improper Coeffs sections
            for coeff_type in switcher_coeffs:
                if switcher_coeffs.get(coeff_type)[0] == 0:
                    continue
                out.write('{}\n'.format(coeff_type))
                out.write('\n')
                counter = 0
                for dic in dicts:
                    convertdict = {}
                    for fields in dic.get(coeff_type):
                        counter += 1
                        convertdict[fields[0]] = counter
                        fields[0] = counter
                        if coeff_type == 'Dihedral Coeffs':
                            out.write('{:>8} {:>10} {:>10} {:>10} {:>10}\n'.format(*fields))
                        elif coeff_type == 'Improper Coeffs':
                            out.write('{:>8} {:>10} {:>10} {:>10}\n'.format(*fields))
                        else:
                            out.write('{:>8} {:>10} {:>10}\n'.format(*fields))
                    switcher_coeffs.get(coeff_type)[1].append(convertdict)
                out.write('\n')

            # Atom section
            out.write('Atoms\n')
            out.write('\n')
            atom_counter = 0
            chain_counter = 0
            for index, dic in enumerate(dicts):
                for num in range(dic.get('Num')):
                    chain_counter += 1
                    for fields in dic.get('Atoms'):
                        atom_counter += 1
                        new_x = r[atom_counter-1][0]
                        new_y = r[atom_counter-1][1]
                        new_z = r[atom_counter-1][2]
                        new_atomtype = atomconvertdicts[index][fields[2]]
                        out.write('{:>8} {:>7} {:>3} {:>12} {:>10} {:>10} {:>10}\n'.format(atom_counter, chain_counter, new_atomtype, fields[3], new_x, new_y, new_z))
            out.write('\n')
            
            # Bond, Angle, Dihedral, and Improper sections
            for section_type in switcher_main:
                if switcher_main.get(section_type)[0] == 0:
                    continue
                out.write('{}\n'.format(section_type))
                out.write('\n')
                atom_counter = 0
                type_counter = 0
                for index, dic in enumerate(dicts):
                    for num in range(dic.get('Num')):
                        for fields in dic.get(section_type):
                            new_id = int(fields[0]) + type_counter
                            section_convertdicts = switcher_main.get(section_type)[1]
                            new_type = section_convertdicts[index][fields[1]]
                            new_atom1 = int(fields[2]) + atom_counter
                            new_atom2 = int(fields[3]) + atom_counter
                            out.write('{:>8} {:>8} {:>6} {:>6}'.format(new_id, new_type, new_atom1, new_atom2))
                            if not section_type == 'Bonds':
                                new_atom3 = int(fields[4]) + atom_counter
                                out.write(' {:>6}'.format(new_atom3))
                                if not section_type == 'Angles':
                                    new_atom4 = int(fields[5]) + atom_counter
                                    out.write(' {:>6}'.format(new_atom4))
                            out.write('\n')
                        atom_counter += len(dic.get('Atoms'))
                        type_counter += len(dic.get(section_type))
                out.write('\n')
