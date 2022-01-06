import numpy as np
import pandas as pd
import psp.simulated_annealing as an
import math
import os
from LigParGen import Converter
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from rdkit import RDLogger
from scipy.spatial.distance import cdist
import psp.MD_lib as MDlib
import glob

RDLogger.DisableLog('rdApp.*')

# OpenBabel setup
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)


# This function try to create a directory
def build_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def is_nan(x):
    return x != x


def len_digit_number(n):
    if n > 0:
        digits = int(math.log10(n)) + 1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n)) + 2  # +1 if you don't count the '-'
    return digits


# This function minimize molecule using UFF forcefield and Steepest Descent method
# INPUT: ID, path and name of XYZ file, row indices of dummy and connecting atoms, name of working directory
# OUTPUT: XYZ coordinates of the optimized molecule
def localopt(unit_name, file_name, dum1, dum2, atom1, atom2, xyz_tmp_dir):
    constraints = ob.OBFFConstraints()
    obConversion.SetInAndOutFormats("xyz", "xyz")
    obConversion.ReadFile(mol, file_name)
    for atom_id in [dum1 + 1, dum2 + 1, atom1 + 1, atom2 + 1]:
        constraints.AddAtomConstraint(atom_id)

    # Set the constraints
    ff.Setup(mol, constraints)
    ff.SteepestDescent(5000)
    ff.UpdateCoordinates(mol)
    obConversion.WriteFile(mol, xyz_tmp_dir + unit_name + '_opt.xyz')

    # Check Connectivity
    neigh_atoms_info_old = connec_info(file_name)
    neigh_atoms_info_new = connec_info(xyz_tmp_dir + unit_name + '_opt.xyz')
    for row in neigh_atoms_info_old.index.tolist():
        if sorted(neigh_atoms_info_old.loc[row]['NeiAtom']) != sorted(
            neigh_atoms_info_new.loc[row]['NeiAtom']
        ):
            unit_opt = pd.read_csv(
                file_name, header=None, skiprows=2, delim_whitespace=True
            )
            return unit_opt
            # print(unit_name, ": Not optimized using steepest descent.")
        else:
            # read XYZ file: skip the first two rows
            unit_opt = pd.read_csv(
                xyz_tmp_dir + unit_name + '_opt.xyz',
                header=None,
                skiprows=2,
                delim_whitespace=True,
            )
            return unit_opt


def gen_dimer_smiles(dum1, dum2, atom1, atom2, input_smiles):
    input_mol = Chem.MolFromSmiles(input_smiles)
    edit_m1 = Chem.EditableMol(input_mol)
    edit_m2 = Chem.EditableMol(input_mol)

    edit_m1.RemoveAtom(dum1)
    edit_m2.RemoveAtom(dum2)

    edit_m1_mol = edit_m1.GetMol()
    edit_m2_mol = edit_m2.GetMol()

    if dum1 < atom1:
        first_atom = atom1 - 1
    else:
        first_atom = atom1

    if dum2 < atom2:
        second_atom = atom2 - 1
    else:
        second_atom = atom2 + edit_m1_mol.GetNumAtoms()

    combo = Chem.CombineMols(edit_m1_mol, edit_m2_mol)
    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(first_atom, second_atom, order=Chem.rdchem.BondType.SINGLE)
    combo_mol = edcombo.GetMol()
    return Chem.MolToSmiles(combo_mol)


def rdkitmol2xyz(unit_name, m, dir_xyz, IDNum):
    try:
        Chem.MolToXYZFile(m, dir_xyz + unit_name + '.xyz', confId=IDNum)
    except Exception:
        obConversion.SetInAndOutFormats("mol", "xyz")
        Chem.MolToMolFile(m, dir_xyz + unit_name + '.mol', confId=IDNum)
        mol = ob.OBMol()
        obConversion.ReadFile(mol, dir_xyz + unit_name + '.mol')
        obConversion.WriteFile(mol, dir_xyz + unit_name + '.xyz')


# This function create XYZ files from SMILES
# INPUT: ID, SMILES, directory name
# OUTPUT: xyz files in 'work_dir', result = DONE/NOT DONE, mol without Hydrogen atom
def smiles_xyz(unit_name, SMILES, dir_xyz):
    try:
        # Get mol(m1) from smiles
        m1 = Chem.MolFromSmiles(SMILES)

        # Add H
        m2 = Chem.AddHs(m1)

        # Get 2D coordinates
        AllChem.Compute2DCoords(m2)

        # Make 3D mol
        AllChem.EmbedMolecule(m2)

        # Change title
        m2.SetProp("_Name", unit_name + '   ' + SMILES)

        # Optimize 3D str
        AllChem.UFFOptimizeMolecule(m2, maxIters=200)
        rdkitmol2xyz(unit_name, m2, dir_xyz, -1)

        result = 'DONE'
    except Exception:
        result, m1 = 'NOT_DONE', ''
    return result, m1


# Search a good conformer
# INPUT: ID, mol without Hydrogen atom, row indices of dummy and connecting atoms, directory
# OUTPUT: XYZ coordinates of the optimized molecule
def find_best_conf(unit_name, m1, dum1, dum2, atom1, atom2, xyz_in_dir):
    m2 = Chem.AddHs(m1)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=100)
    cid_list = []
    for cid in cids:
        AllChem.UFFOptimizeMolecule(m2, confId=cid)
        conf = m2.GetConformer(cid)
        ffu = AllChem.UFFGetMoleculeForceField(m2, confId=cid)
        cid_list.append(
            [
                cid,
                abs(
                    Chem.rdMolTransforms.GetDihedralDeg(
                        conf, int(dum1), int(atom1), int(atom2), int(dum2)
                    )
                ),
                ffu.CalcEnergy(),
            ]
        )
    cid_list = pd.DataFrame(cid_list, columns=['cid', 'Dang', 'Energy'])
    cid_list = cid_list.sort_values(by=['Dang'], ascending=False)
    cid_list = cid_list[
        cid_list['Dang'] > int(cid_list.head(1)['Dang'].values[0]) - 8.0
    ]
    cid_list = cid_list.sort_values(by=['Energy'], ascending=True)
    rdkitmol2xyz(unit_name, m2, xyz_in_dir, int(cid_list.head(1)['cid'].values[0]))


# This function indentifies row numbers of dummy atoms
# INPUT: SMILES
# OUTPUT: row indices of dummy atoms and nature of bond with connecting atom
def FetchDum(smiles):
    m = Chem.MolFromSmiles(smiles)
    dummy_index = []
    if m is not None:
        for atom in m.GetAtoms():
            if atom.GetSymbol() == '*':
                dummy_index.append(atom.GetIdx())
        for bond in m.GetBonds():
            if (
                bond.GetBeginAtom().GetSymbol() == '*'
                or bond.GetEndAtom().GetSymbol() == '*'
            ):
                bond_type = bond.GetBondType()
                break
    return dummy_index, str(bond_type)


# Build an Oligomer
# INPUT: ID, Length of an oligomer to be built, XYZ-coordinates of a monomer,
# row numbers for dummy and connecting atoms, unit distance
# OUTPUT: XYZ-coordinates of an oligomer and outcome of connectivity check
def build(unit_name, length, unit, dum1, dum2, atom1, atom2, unit_dis):
    add_dis = add_dis_func(unit, atom1, atom2)

    # XYZ positions of atom2
    unit_disX = unit.loc[atom2].values[1]
    unit_disY = unit.loc[atom2].values[2]
    unit_disZ = unit.loc[atom2].values[3] + unit_dis + add_dis

    build = pd.DataFrame()  # START A DATAFRAME
    for len in np.arange(1, length + 1, 1):  # EACH UNIT IN A MOLECULE
        # First unit of the oligomer
        if len == 1:
            build = build.append(unit.drop([dum2]), ignore_index=True)

        # Last unit of the Oligomer
        elif len == length:
            unit[1] = unit[1] + unit_disX
            unit[2] = unit[2] + unit_disY
            unit[3] = unit[3] + unit_disZ

            first_unit = build.copy()
            second_unit = unit.drop([dum1]).copy()
            build = build.append(unit.drop([dum1]), ignore_index=True)
            check_connectivity = CheckConnectivity(
                unit_name, first_unit, second_unit, build, atom2
            )
            if check_connectivity == 'CORRECT':
                # Calculate distance between atoms in first_unit and second_unit
                dist = cdist(
                    first_unit[[1, 2, 3]].values, second_unit[[1, 2, 3]].values
                )
                # Find out index number for atom2 and atom1 in first_unit and second_unit
                if dum1 > atom1:
                    atom1_index = atom1
                else:
                    atom1_index = atom1 - 1
                if dum2 > atom2:
                    atom2_index = atom2
                else:
                    atom2_index = atom2 - 1
                # Add some distance to avoid detection atom2-atom1 bond in dimer
                dist[atom2_index, atom1_index] += 100.0

                if (dist < 1.0).any():  # in angstrom
                    check_connectivity = 'WRONG'

        # Any middle unit of the Oligomer
        else:
            unit[1] = unit[1] + unit_disX
            unit[2] = unit[2] + unit_disY
            unit[3] = unit[3] + unit_disZ
            build = build.append(unit.drop([dum1, dum2]), ignore_index=True)

    return build, check_connectivity


# Build all possible dimers (For DIMER method)
# INPUT: ID, XYZ-coordinates of two monomers, row numbers for dummy and connecting atoms, unit distance
# OUTPUT: XYZ-coordinates of dimer, Connectivity check, and row numbers for new dummy and connecting atoms
def TwoMonomers_Dimer(unit_name, unit1, unit2, dum1, dum2, atom1, atom2, dum, unit_dis):
    add_dis = add_dis_func(unit1, atom1, atom2)

    unit1 = trans_origin(unit1, dum1)
    unit1 = alignZ(unit1, dum1, dum2)
    unit2 = trans_origin(unit2, dum1)
    unit2 = alignZ(unit2, dum1, dum2)
    # XYZ positions of atom2
    unit_disX = unit1.loc[atom2].values[1]
    unit_disY = unit1.loc[atom2].values[2]
    unit_disZ = unit1.loc[atom2].values[3] + unit_dis + add_dis

    build = pd.DataFrame()  # START A DATAFRAME
    build = build.append(unit1.drop([dum2]), ignore_index=True)

    # Move the second unit
    unit2[1] = unit2[1] + unit_disX
    unit2[2] = unit2[2] + unit_disY
    unit2[3] = unit2[3] + unit_disZ

    first_unit = build.copy()
    second_unit = unit2.drop([dum1]).copy()
    build = build.append(unit2.drop([dum1]), ignore_index=True)
    check_connectivity = CheckConnectivity(
        unit_name, first_unit, second_unit, build, atom2
    )
    if check_connectivity == 'CORRECT':
        # Calculate distance between atoms in first_unit and second_unit
        dist = cdist(first_unit[[1, 2, 3]].values, second_unit[[1, 2, 3]].values)
        # Find out index number for atom2 and atom1 in first_unit and second_unit
        if dum1 > atom1:
            atom1_index = atom1
        else:
            atom1_index = atom1 - 1
        if dum2 > atom2:
            atom2_index = atom2
        else:
            atom2_index = atom2 - 1
        # Add some distance to avoid detection atom2-atom1 bond in dimer
        dist[atom2_index, atom1_index] += 100.0

        if (dist < 1.6).any():  # in angstrom
            check_connectivity = 'WRONG'

        dum1_2nd, atom1_2nd, dum2_2nd, atom2_2nd = (
            dum1,
            atom1,
            dum2 + unit1.shape[0] - 2,
            atom2 + unit1.shape[0] - 2,
        )

    else:
        dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd = 0, 0, 0, 0

    return build, check_connectivity, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd


# Find row number of atoms associated with an atom
# INPUT: row number of atoms (atoms) and connecting information obtained from Openbabel
# OUTPUT: atoms directly connected with each atom
def findConnectingAtoms(atoms, neigh_atoms_info):
    ConnectingAtoms = []
    for i in atoms:
        ConnectingAtoms.extend(neigh_atoms_info.loc[i]['NeiAtom'])
    return ConnectingAtoms


# Find bond order between two atoms
# INPUT: Row number of atom1 and atom2, and connectivity information obtained from Openbabel
# OUTPUT: Bond order between atom1 and atom2
def find_bondorder(atom1, rot_atom1, neigh_atoms_info):
    index_rot_atom1 = neigh_atoms_info.loc[atom1]['NeiAtom'].index(rot_atom1)
    return neigh_atoms_info.loc[atom1]['BO'][index_rot_atom1]


# Find single bonds and associated atoms
# INPUT: unit_name, XYZ coordinates, xyz_tmp_dir
# OUTPUT: mol for RDKit
def xyz2RDKitmol(unit_name, unit, xyz_tmp_dir):
    obConversion.SetInAndOutFormats("xyz", "mol")
    gen_xyz(xyz_tmp_dir + unit_name + '.xyz', unit)
    obConversion.ReadFile(mol, xyz_tmp_dir + unit_name + '.xyz')
    obConversion.WriteFile(mol, xyz_tmp_dir + unit_name + '.mol')
    return Chem.MolFromMolFile(xyz_tmp_dir + unit_name + '.mol')


# Find single bonds and associated atoms
# INPUT: unit_name, XYZ coordinates, xyz_tmp_dir
# OUTPUT: List of atoms with bond_order = 1
def single_bonds(unit_name, unit, xyz_tmp_dir):
    try:
        mol = xyz2RDKitmol(unit_name, unit, xyz_tmp_dir)
        RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        single_bond = pd.DataFrame(mol.GetSubstructMatches(RotatableBond))

    except Exception:
        single_bond = []
        gen_xyz(xyz_tmp_dir + unit_name + '.xyz', unit)
        neigh_atoms_info = connec_info(xyz_tmp_dir + unit_name + '.xyz')
        for index, row in neigh_atoms_info.iterrows():
            if len(row['NeiAtom']) < 2:
                neigh_atoms_info = neigh_atoms_info.drop(index)

        for index, row in neigh_atoms_info.iterrows():
            for j in row['NeiAtom']:
                bond_order = find_bondorder(index, j, neigh_atoms_info)
                if bond_order == 1:
                    single_bond.append([index, j])
        single_bond = pd.DataFrame(single_bond)
        H_list = list(unit.loc[unit[0] == 'H'].index)
        F_list = list(unit.loc[unit[0] == 'F'].index)
        Cl_list = list(unit.loc[unit[0] == 'Cl'].index)
        Br_list = list(unit.loc[unit[0] == 'Br'].index)
        I_list = list(unit.loc[unit[0] == 'I'].index)
        remove_list = H_list + F_list + Cl_list + Br_list + I_list

        single_bond = single_bond.loc[~single_bond[0].isin(remove_list)]
        single_bond = single_bond.loc[~single_bond[1].isin(remove_list)]
        single_bond = pd.DataFrame(np.sort(single_bond.values)).drop_duplicates()
    return single_bond


# This function collects row numbers of group of atoms (directly or indirectly) connected to an atom
# INPUT: Row number of an atom (atom), XYZ-coordinates of a molecule (unit), and
# connectivity information obtained from Openbabel
# OUTPUT: Row numbers of rotating atoms
def findrotgroups(atom, unit, neigh_atoms_info):
    nei_atom = neigh_atoms_info.loc[atom]['NeiAtom']
    neinei_atom = {}
    for i in nei_atom:
        neinei_atom[i] = neigh_atoms_info.loc[i][
            'NeiAtom'
        ].copy()  # collect all neighboring atoms for each atom connected to connecting atom 1
        while len(neinei_atom[i]) < unit.values.shape[0]:
            NumConnect = neinei_atom[i].copy()
            if atom in neinei_atom[i]:
                neinei_atom[i].remove(atom)
            neinei_atom[i].extend(
                list(set(findConnectingAtoms(neinei_atom[i], neigh_atoms_info)))
            )
            if i in neinei_atom[i]:
                neinei_atom[i].remove(i)
            neinei_atom[i] = list(set(neinei_atom[i]))
            if sorted(neinei_atom[i]) == sorted(NumConnect):
                break
    return neinei_atom  # rot_groups


# Find out row numbers of atoms which are allowed to rotate
# INPUT: Row numbers of the two rotating atoms, and
# groups of atoms connected (directly or indirectly) to the first rotating atom
# OUTPUT: Row numbers of atoms allowed to rotate
def search_rot_atoms(atom1, rot_atom1, rot_groups):
    rot_atoms = []  # collect all atoms to be rotated
    for i in list(rot_groups.keys()):
        if i != rot_atom1:
            rot_atoms.extend(rot_groups[i])
    rot_atoms = list(set([atom1, rot_atom1] + rot_atoms + list(rot_groups.keys())))
    return rot_atoms


# Connection information obtained by OpenBabel
# INPUT: XYZ file
# OUTPUT: Connectivity information
def connec_info(unit_name):
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("xyz")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, unit_name)
    neigh_atoms_info = []

    for atom in ob.OBMolAtomIter(mol):
        neigh_atoms = []
        bond_orders = []
        for allatom in ob.OBAtomAtomIter(atom):
            neigh_atoms.append(allatom.GetIndex())
            bond_orders.append(atom.GetBond(allatom).GetBondOrder())
        neigh_atoms_info.append([neigh_atoms, bond_orders])
    neigh_atoms_info = pd.DataFrame(neigh_atoms_info, columns=['NeiAtom', 'BO'])
    return neigh_atoms_info


# This function generates a xyz file
# INPUT: Name of a output file and a DataFrame of element names and respective XYZ-coordinates
# OUTPUT: Write a XYZ file
def gen_xyz(filename, unit):
    unit.iloc[:, 1:4] = unit.iloc[:, 1:4].round(6)
    with open(filename, 'w') as f:
        f.write(str(unit.values.shape[0]))  # NUMBER OF ATOMS
        f.write("\n\n")  # TWO SPACES
        unit.to_csv(
            f, sep=' ', index=False, header=False
        )  # XYZ COORDINATES OF NEW MOLECULE


# This function generates a VASP input (polymer) file
# INPUT: name of VASP directory, name of a monomer, XYZ-coordinates, row numbers for dummy and
# connecting atoms , chemical name of dummy atom, Serial number
# OUTPUT: Generates a VASP input file
def gen_vasp(
    vasp_dir,
    unit_name,
    unit,
    dum1,
    dum2,
    atom1,
    atom2,
    dum,
    unit_dis,
    SN=0,
    length=0,
    Inter_Chain_Dis=12,
    Polymer=False,
):
    add_dis = add_dis_func(unit, atom1, atom2)

    unit = trans_origin(unit, atom2)
    unit = alignZ(unit, atom2, dum1)
    unit = unit.sort_values(by=[0])

    if SN == 0 and length == 0:
        file_name = vasp_dir + unit_name.replace('.xyz', '') + '.vasp'
    elif SN == 0 and length != 0:
        file_name = (
            vasp_dir + unit_name.replace('.xyz', '') + '_N' + str(length) + '.vasp'
        )
    elif SN != 0 and length == 0:
        file_name = vasp_dir + unit_name.replace('.xyz', '') + '_C' + str(SN) + '.vasp'
    else:
        file_name = (
            vasp_dir
            + unit_name.replace('.xyz', '')
            + '_N'
            + str(length)
            + '_C'
            + str(SN)
            + '.vasp'
        )

    file = open(file_name, 'w+')
    file.write('### ' + str(unit_name) + ' ###\n')
    file.write('1\n')

    # Get the size of the box
    a_vec = unit[1].max() - unit[1].min() + Inter_Chain_Dis
    b_vec = unit[2].max() - unit[2].min() + Inter_Chain_Dis

    if Polymer:
        c_vec = unit.loc[dum1][3] + unit_dis + add_dis  #
    else:
        c_vec = unit[3].max() - unit[3].min() + Inter_Chain_Dis

    # move unit to the center of a box
    unit[1] = unit[1] - unit[1].min() + Inter_Chain_Dis / 2
    unit[2] = unit[2] - unit[2].min() + Inter_Chain_Dis / 2

    if Polymer:
        unit[3] = unit[3] + (1.68 + unit_dis + add_dis) / 2
    else:
        unit[3] = unit[3] - unit[3].min() + Inter_Chain_Dis / 2

    unit = unit.drop([dum1, dum2])
    file.write(' ' + str(a_vec) + ' ' + str(0.0) + ' ' + str(0.0) + '\n')
    file.write(' ' + str(0.0) + ' ' + str(b_vec) + ' ' + str(0.0) + '\n')
    file.write(' ' + str(0.0) + ' ' + str(0.0) + ' ' + str(c_vec) + '\n')

    ele_list = []
    count_ele_list = []
    for element in sorted(set(unit[0].values)):
        ele_list.append(element)
        count_ele_list.append(list(unit[0].values).count(element))

    for item in ele_list:
        file.write(str(item) + '  ')

    file.write('\n ')
    for item in count_ele_list:
        file.write(str(item) + ' ')

    file.write('\nCartesian\n')

    file.write(unit[[1, 2, 3]].to_string(header=False, index=False))
    file.close()


# Distance between two points
def distance(x1, x2, x3, y1, y2, y3):
    return np.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2 + (x3 - y3) ** 2)


# Angle between two vectors (Vector1: a1,a2; Vector2: b1,b2)
# INPUT: 4 data points with XYZ coordinates
# OUTPUT: Angle in Degree
def angle_vec(a1, a2, b1, b2):
    x1, x2, x3 = a2[1] - a1[1], a2[2] - a1[2], a2[3] - a1[3]
    y1, y2, y3 = b2[1] - b1[1], b2[2] - b1[2], b2[3] - b1[3]
    ab = x1 * y1 + x2 * y2 + x3 * y3
    mod_a = np.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
    mod_b = np.sqrt(y1 * y1 + y2 * y2 + y3 * y3)
    alpha = math.acos(np.around(ab / (mod_a * mod_b), decimals=15))
    return alpha * (180 / math.pi)


# Translation to origin
# INPUT: XYZ-coordinates and row number of an atom which will be moved to the origin.
# OUTPUT: A new sets of XYZ-coordinates
def trans_origin(unit, atom1):  # XYZ coordinates and angle
    unit[1] = unit[1] - (unit.iloc[atom1][1])
    unit[2] = unit[2] - (unit.iloc[atom1][2])
    unit[3] = unit[3] - (unit.iloc[atom1][3])
    return unit


# Align a molecule on Z-axis wrt two atoms
# INPUT: XYZ-coordinates, row numbers of two atoms
# OUTPUT: A new sets of XYZ-coordinates
def alignZ(unit, atom1, atom2):
    dis_zx = np.sqrt(
        (unit.iloc[atom1].values[3] - unit.iloc[atom2].values[3]) ** 2
        + (unit.iloc[atom1].values[1] - unit.iloc[atom2].values[1]) ** 2
    )
    angle_zx = (np.arccos(unit.iloc[atom2].values[3] / dis_zx)) * 180.0 / np.pi
    if unit.iloc[atom2].values[1] > 0.0:  # or angle_zx < 90.0: # check and improve
        angle_zx = -angle_zx
    unit = rotateXZ(unit, angle_zx)

    dis_zy = np.sqrt(
        (unit.iloc[atom1].values[3] - unit.iloc[atom2].values[3]) ** 2
        + (unit.iloc[atom1].values[2] - unit.iloc[atom2].values[2]) ** 2
    )
    angle_zy = (np.arccos(unit.iloc[atom2].values[3] / dis_zy)) * 180.0 / np.pi
    if unit.iloc[atom2].values[2] > 0.0:  # or angle_zy < 90.0: # need to improve
        angle_zy = -angle_zy

    unit = rotateYZ(unit, angle_zy)
    return unit


# Move molecule
def move_barycenter(unit, xyz_shift, origin=True, barycenter=True):
    unit_copy = unit.copy()
    if origin is True:
        if barycenter is False:
            unit_copy[1] = unit_copy[1] - unit_copy.min()[1]
            unit_copy[2] = unit_copy[2] - unit_copy.min()[2]
            unit_copy[3] = unit_copy[3] - unit_copy.min()[3]
        else:
            unit_copy[1] = unit_copy[1] - unit_copy.mean()[1]
            unit_copy[2] = unit_copy[2] - unit_copy.mean()[2]
            unit_copy[3] = unit_copy[3] - unit_copy.mean()[3]
    else:
        unit_copy[1] = unit_copy[1] + xyz_shift[0]
        unit_copy[2] = unit_copy[2] + xyz_shift[1]
        unit_copy[3] = unit_copy[3] + xyz_shift[2]
    return unit_copy


# Rotate in X, Y and Z directions simultaneously
def rotateXYZ(unit, theta3, theta2, theta1):
    th1 = theta1 * np.pi / 180.0  # Z-axis
    th2 = theta2 * np.pi / 180.0  # Y-axis
    th3 = theta3 * np.pi / 180.0  # X-axis
    Rot_matrix = np.array(
        [
            [
                np.cos(th1) * np.cos(th2),
                np.cos(th1) * np.sin(th2) * np.sin(th3) - np.sin(th1) * np.cos(th3),
                np.cos(th1) * np.sin(th2) * np.cos(th3) + np.sin(th1) * np.sin(th3),
            ],
            [
                np.sin(th1) * np.cos(th2),
                np.sin(th1) * np.sin(th2) * np.sin(th3) + np.cos(th1) * np.cos(th3),
                np.sin(th1) * np.sin(th2) * np.cos(th3) - np.cos(th1) * np.sin(th3),
            ],
            [-np.sin(th2), np.cos(th2) * np.sin(th3), np.cos(th2) * np.cos(th3)],
        ]
    )

    rot_XYZ = unit.loc[:, [1, 2, 3]].copy()
    rotated_unit = rot_XYZ.values.dot(Rot_matrix)
    newXYZ = pd.DataFrame(rotated_unit, columns=[1, 2, 3])
    newXYZ.index = unit.index
    unit.loc[:, [1, 2, 3]] = newXYZ.loc[:, [1, 2, 3]]
    return unit


# Rotate in X, Y and Z directions simultaneously
def rotateXYZOrigin(unit_copy, theta1, theta2, theta3):
    x_move = unit_copy.mean()[1]
    y_move = unit_copy.mean()[2]
    z_move = unit_copy.mean()[3]
    unit_mod = move_barycenter(
        unit_copy, [-x_move, -y_move, -z_move], origin=False, barycenter=False
    )
    unit_mod = rotateXYZ(unit_mod, theta1, theta2, theta3)
    unit_mod = move_barycenter(
        unit_mod, [x_move, y_move, z_move], origin=False, barycenter=False
    )
    return unit_mod


# Rotate Molecule along Z-axis
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateZ(unit, theta, rot_atoms):  # XYZ coordinates, angle and atoms to be rotated
    R_z = np.array(
        [
            [np.cos(theta * np.pi / 180.0), -np.sin(theta * np.pi / 180.0), 0],
            [np.sin(theta * np.pi / 180.0), np.cos(theta * np.pi / 180.0), 0],
            [0, 0, 1],
        ]
    )
    rot_XYZ = unit.loc[(rot_atoms), [1, 2, 3]].copy()

    XYZcollect = []
    for eachatom in np.arange(rot_XYZ.values.shape[0]):
        rotate_each = rot_XYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ = pd.DataFrame(XYZcollect, index=rot_atoms, columns=[1, 2, 3])
    unit.loc[(rot_atoms), [1, 2, 3]] = newXYZ[[1, 2, 3]]
    return unit


# Rotate on XZ plane
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateXZ(unit, theta):  # XYZ coordinates and angle
    R_z = np.array(
        [
            [np.cos(theta * np.pi / 180.0), -np.sin(theta * np.pi / 180.0)],
            [np.sin(theta * np.pi / 180.0), np.cos(theta * np.pi / 180.0)],
        ]
    )
    oldXYZ = unit[[1, 3]].copy()
    XYZcollect = []
    for eachatom in np.arange(oldXYZ.values.shape[0]):
        rotate_each = oldXYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ = pd.DataFrame(XYZcollect)
    unit[[1, 3]] = newXYZ[[0, 1]]
    return unit


# Rotate on ZY plane
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateYZ(unit, theta):  # XYZ coordinates and angle
    R_z = np.array(
        [
            [np.cos(theta * np.pi / 180.0), -np.sin(theta * np.pi / 180.0)],
            [np.sin(theta * np.pi / 180.0), np.cos(theta * np.pi / 180.0)],
        ]
    )
    oldXYZ = unit[[2, 3]].copy()
    XYZcollect = []
    for eachatom in np.arange(oldXYZ.values.shape[0]):
        rotate_each = oldXYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ = pd.DataFrame(XYZcollect)
    unit[[2, 3]] = newXYZ[[0, 1]]
    return unit


# Check connectivity between atoms in a dimer (Only for unit1 == unit2)
# INPUT: ID, XYZ-coordinates of unit1, unit2 and dimer.
# OUTPUT: Check connectivity (CORRECT or WRONG)
def CheckConnectivity(unit_name, unit1, unit2, dimer, connected_atom):
    gen_xyz('work_dir/unit1_' + unit_name + '.xyz', unit1)
    gen_xyz('work_dir/unit2_' + unit_name + '.xyz', unit2)
    gen_xyz('work_dir/dimer_' + unit_name + '.xyz', dimer)

    neigh_atoms_info_unit1 = connec_info('work_dir/unit1_' + unit_name + '.xyz')
    neigh_atoms_info_unit2 = connec_info('work_dir/unit2_' + unit_name + '.xyz')
    neigh_atoms_info_dimer = connec_info('work_dir/dimer_' + unit_name + '.xyz')

    Num_atoms_unit1 = len(neigh_atoms_info_unit1.index.tolist())
    for index, row in neigh_atoms_info_unit2.iterrows():
        row['NeiAtom'] = [x + Num_atoms_unit1 for x in row['NeiAtom']]
    neigh_atoms_info_ideal_dimer = pd.concat(
        [neigh_atoms_info_unit1, neigh_atoms_info_unit2]
    )

    check_connectivity = 'CORRECT'

    list1_atoms = neigh_atoms_info_dimer.iloc[connected_atom]['NeiAtom']
    list1_atoms_ideal = neigh_atoms_info_ideal_dimer.iloc[connected_atom]['NeiAtom']

    # Connected more atoms: WRONG
    if len(list1_atoms) - len(list1_atoms_ideal) != 1:
        return 'WRONG'

    # Connected to different atoms: WRONG
    if (
        len(
            list(
                set(list1_atoms)
                .intersection(set(list1_atoms_ideal))
                .symmetric_difference(set(list1_atoms_ideal))
            )
        )
        != 0
    ):
        return 'WRONG'

    second_connected_atom = list(
        set(list1_atoms).symmetric_difference(set(list1_atoms_ideal))
    )[0]
    list2_atoms = neigh_atoms_info_dimer.iloc[second_connected_atom]['NeiAtom']
    list2_atoms_ideal = neigh_atoms_info_ideal_dimer.iloc[second_connected_atom][
        'NeiAtom'
    ]

    # Connected more atoms: WRONG
    if len(list2_atoms) - len(list2_atoms_ideal) != 1:
        return 'WRONG'

    # Connected to different atoms: WRONG
    if (
        len(
            list(
                set(list2_atoms)
                .intersection(set(list2_atoms_ideal))
                .symmetric_difference(set(list2_atoms_ideal))
            )
        )
        != 0
    ):
        return 'WRONG'

    # Other atoms are connected: WRONG
    for row in range(len(neigh_atoms_info_ideal_dimer.index.tolist())):
        if row not in [connected_atom, second_connected_atom]:
            if sorted(neigh_atoms_info_ideal_dimer.iloc[row]['NeiAtom']) != sorted(
                neigh_atoms_info_dimer.iloc[row]['NeiAtom']
            ):
                return 'WRONG'

    return check_connectivity


# This function rotate a molecule; translate to origin, align on the Z-axis, rotate around Z-axis
# INPUT: XYZ coordinate, row numbers of two atoms, Atoms involve in rotation, angle
# OUTPUT: Rotated geometry
def rot_unit(unit, atom1, rot_atom1, rot_atoms, angle):
    unit = trans_origin(unit, atom1)
    unit = alignZ(unit, atom1, rot_atom1)
    unit = rotateZ(unit, angle, rot_atoms)
    return unit


# This function estimate distance between two repeating units
# INPUT: XYZ coordinates, row numbers for connecting atoms
# OUTPUT: Distance
def add_dis_func(unit, atom1, atom2):
    add_dis = 0.0
    if unit.loc[atom1][0] == 'C' and unit.loc[atom2][0] == 'N':
        add_dis = -0.207
    elif unit.loc[atom1][0] == 'N' and unit.loc[atom2][0] == 'N':
        add_dis = -0.4
    elif unit.loc[atom1][0] == 'C' and unit.loc[atom2][0] == 'O':
        add_dis = -0.223
    elif unit.loc[atom1][0] == 'O' and unit.loc[atom2][0] == 'O':
        add_dis = -0.223
    return add_dis


# Rotate over Z-axis and combine it with another unit to build new monomer unit (non periodic)
def build_dimer_rotate(
    unit_name, rot_angles, unit1, unit2, dum, dum1, dum2, atom1, atom2, unit_dis
):
    add_dis = add_dis_func(unit1, atom1, atom2)
    list_conf = []
    unit_2nd = {}  # All possible new units
    count = 1
    unit_dimer = pd.DataFrame()

    unit1 = trans_origin(unit1, atom1)
    unit1 = alignZ(unit1, atom1, dum1)

    for i in rot_angles:
        unit_2nd[count] = unit2.copy()
        unit_2nd[count] = trans_origin(unit_2nd[count], atom1)
        unit_2nd[count] = alignZ(unit_2nd[count], atom1, atom2)
        unit_2nd[count] = rotateZ(
            unit_2nd[count], i, np.arange(len(unit_2nd[count][0].values))
        )
        # combine two units
        unit_2nd[count] = trans_origin(unit_2nd[count], dum2)
        unit_2nd[count] = alignZ(unit_2nd[count], dum2, atom2)
        unit_2nd[count] = trans_origin(unit_2nd[count], atom2)
        unit_2nd[count][3] = unit_2nd[count][3] + unit1[3][dum1] + unit_dis + add_dis
        unit_2nd[count] = unit_2nd[count].drop([dum2])
        unit_2nd[count] = unit_2nd[count].append(unit1.drop([dum1]), ignore_index=True)

        gen_xyz('work_dir/' + unit_name + '.xyz', unit_2nd[count])
        neigh_atoms_info_dimer = connec_info('work_dir/' + unit_name + '.xyz')

        dum1_2nd, atom1_2nd, dum2_2nd, atom2_2nd = (
            dum2 + unit1.shape[0] - 2,
            atom2 + unit1.shape[0] - 2,
            dum1,
            atom1,
        )

        Num_atoms_unit = len(unit_2nd[count].index.tolist())
        check_connectivity = CheckConnectivity(
            unit_name,
            unit_2nd[count].iloc[0: int(Num_atoms_unit / 2)],
            unit_2nd[count].iloc[int(Num_atoms_unit / 2): Num_atoms_unit],
            unit_2nd[count],
            atom2,
        )

        if check_connectivity == 'CORRECT':
            # Distance between two dummy atoms and angle between two vectors originated from dummy and connecting atoms
            dis_dum1_dum2 = distance(
                unit_2nd[count][1][dum1_2nd],
                unit_2nd[count][2][dum1_2nd],
                unit_2nd[count][3][dum1_2nd],
                unit_2nd[count][1][dum2_2nd],
                unit_2nd[count][2][dum2_2nd],
                unit_2nd[count][3][dum2_2nd],
            )
            ang_1st_2nd = angle_vec(
                unit_2nd[count].loc[atom1_2nd],
                unit_2nd[count].loc[dum1_2nd],
                unit_2nd[count].loc[atom2_2nd],
                unit_2nd[count].loc[dum2_2nd],
            )

            list_conf.append([count, dis_dum1_dum2, ang_1st_2nd])

        count += 1
    try:
        list_conf = pd.DataFrame(list_conf, columns=['count', 'dis', 'Dang'])
        list_conf = list_conf.sort_values(by=['Dang', 'dis'], ascending=False)

        unit_dimer = unit_2nd[list_conf['count'].head(1).values[0]].copy()

        # Rearrange rows
        rows = unit_dimer.index.tolist()
        for i in [dum1_2nd, atom1_2nd, atom2_2nd, dum2_2nd]:
            rows.remove(i)
        new_rows = [dum1_2nd, atom1_2nd, atom2_2nd, dum2_2nd] + rows
        unit_dimer = unit_dimer.loc[new_rows].reset_index(drop=True)
        dum1_2nd, atom1_2nd, atom2_2nd, dum2_2nd = 0, 1, 2, 3

        gen_xyz('work_dir/' + unit_name + '.xyz', unit_dimer)
        neigh_atoms_info_dimer = connec_info('work_dir/' + unit_name + '.xyz')

    except Exception:
        pass

    return unit_dimer, neigh_atoms_info_dimer, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd


# This function create a conformer
# INPUT: Name of the molecule, step number in SA, XYZ-coordinates, row numbers of two atoms (single bond).
# neighboring atom information, angle, temporary directory name, row numbers of dummy and connecting atoms
# OUTPUT: Name of the XYZ file, XYZ-coordinates of rotated unit, dis_dum1_dum2, ang_1st_2nd
def create_conformer(
    unit_name,
    sl,
    unit,
    bond,
    neigh_atoms_info,
    angle,
    xyz_tmp_dir,
    dum1,
    dum2,
    atom1,
    atom2,
):
    rot_groups = findrotgroups(bond[0], unit, neigh_atoms_info)
    rot_atoms = search_rot_atoms(bond[0], bond[1], rot_groups)
    conf_unit = rot_unit(unit, bond[0], bond[1], rot_atoms, angle)

    # Distance between two dummy atoms and angle between two vectors originated from dummy and connecting atoms
    dis_dum1_dum2 = distance(
        conf_unit[1][dum1],
        conf_unit[2][dum1],
        conf_unit[3][dum1],
        conf_unit[1][dum2],
        conf_unit[2][dum2],
        conf_unit[3][dum2],
    )
    ang_1st_2nd = angle_vec(
        conf_unit.loc[atom1],
        conf_unit.loc[dum1],
        conf_unit.loc[atom2],
        conf_unit.loc[dum2],
    )

    file_name = (
        xyz_tmp_dir
        + unit_name
        + '__'
        + str(sl)
        + '_'
        + str(bond[0])
        + '_'
        + str(bond[1])
        + '_'
        + str(angle)
        + '.xyz'
    )
    gen_xyz(file_name, conf_unit)

    penalty = 0
    neigh_atoms_info_new = connec_info(file_name)
    for row in neigh_atoms_info.index.tolist():
        if sorted(neigh_atoms_info.loc[row]['NeiAtom']) != sorted(
            neigh_atoms_info_new.loc[row]['NeiAtom']
        ):
            penalty = 1
    return file_name, conf_unit, dis_dum1_dum2, ang_1st_2nd, penalty


# Build a dimer and check connectivity
# INPUT: ID, XYZ-coordinates, Connectivity in monomer, row numbers of dummy and connecting atoms, unit distance
# OUTPUT: Connectivity in Dimer
def mono2dimer(
    unit_name,
    unit_input,
    check_connectivity_monomer,
    dum1,
    dum2,
    atom1,
    atom2,
    unit_dis,
):
    if check_connectivity_monomer == 'CORRECT':
        unit_copy = unit_input.copy()
        # Build a dimer
        unit_copy = trans_origin(unit_copy, dum1)
        unit_copy = alignZ(unit_copy, dum1, dum2)

        polymer, check_connectivity_dimer = build(
            unit_name, 2, unit_copy, dum1, dum2, atom1, atom2, unit_dis
        )
    else:
        check_connectivity_dimer = 'WRONG'

    return check_connectivity_dimer


# Build oligomers
# INPUT: XYZ-coordinates, ID, row numbers of dummy and connecting atoms, length of oligomer, unit distance and
# information by OpenBabel
def oligomer_build(
    unit, unit_name, dum1, dum2, atom1, atom2, oligo_len, unit_dis, neigh_atoms_info
):
    unit_copy = unit.copy()
    unit_copy = trans_origin(unit_copy, dum1)
    unit_copy = alignZ(unit_copy, dum1, dum2)
    if oligo_len == 1:
        oligomer = unit_copy
        dum2_oligo = dum2
        atom2_oligo = atom2
    else:
        oligomer, check_connectivity_dimer = build(
            unit_name, oligo_len, unit_copy, dum1, dum2, atom1, atom2, unit_dis
        )

        add_atoms = (oligo_len - 1) * (len(unit.index) - 2)
        dum2_oligo = dum2 + add_atoms
        atom2_oligo = atom2 + add_atoms

    if find_bondorder(dum1, atom1, neigh_atoms_info) == 1:
        dums = [dum1, dum2_oligo]
        atoms = [atom1, atom2_oligo]
        for i, j in zip(dums, atoms):
            vec1, vec2, vec3 = (
                oligomer.loc[i][1] - oligomer.loc[j][1],
                oligomer.loc[i][2] - oligomer.loc[j][2],
                oligomer.loc[i][3] - oligomer.loc[j][3],
            )
            vec_mag = np.sqrt(vec1 * vec1 + vec2 * vec2 + vec3 * vec3)
            vec_normal = (vec1, vec2, vec3) / vec_mag
            new_coord = (
                oligomer.loc[j][1],
                oligomer.loc[j][2],
                oligomer.loc[j][3],
            ) + 1.08 * vec_normal
            oligomer = oligomer.append(
                pd.DataFrame(['H'] + list(new_coord)).T, ignore_index=True
            )
    return oligomer, dum1, atom1, dum2_oligo, atom2_oligo


# This function updates XYZ coordinates in OBmol
# INPUT: XYZ file names of original (reference) and new one
# Updated OBmol (Connectivity info from reference file and XYZ coordinates from new file)
def OBmolUpdateXYZcoordinates(ori_xyz, new_xyz):
    new_unit = pd.read_csv(new_xyz, header=None, skiprows=2, delim_whitespace=True)
    obConversion.ReadFile(mol, ori_xyz)
    for atm in np.arange(new_unit.shape[0]):
        a = mol.GetAtom(int(atm) + 1)
        a.SetVector(new_unit.loc[atm, 1], new_unit.loc[atm, 2], new_unit.loc[atm, 3])
    return mol


# This function compare connectivity between two molecules; If they are same it returns CORRECT otherwise WRONG.
# INPUT: PATH + file name of the first and second XYZ files
# OUTPUT: CORRECT or WRONG
def CompareConnectInfo(first_file, second_file):
    result = 'CORRECT'
    neigh_atoms_info = connec_info(first_file)
    neigh_atoms_info_new = connec_info(second_file)
    for row in neigh_atoms_info.index.tolist():
        if sorted(neigh_atoms_info.loc[row]['NeiAtom']) != sorted(
            neigh_atoms_info_new.loc[row]['NeiAtom']
        ):
            result = 'WRONG'
    return result


# This function stretches a repeating unit by moving a linking atom (+ corresponding dummy atom). It performs constraint
# optimization by fixing positions of linking and dummy atoms using the UFF forcefield and Steepest Descent method
# INPUT: ID, unit (XYZ coordinates), row numbers of dummy and linking atoms, folder name
# OUTPUT: new_unit (XYZ coordinates), PATH + new XYZ file name
def MakePolymerStraight(
    unit_name,
    ref_xyz,
    unit_inp,
    dum1_inp,
    dum2_inp,
    atom1_inp,
    atom2_inp,
    xyz_tmp_dir,
    Tol_ChainCorr,
):
    constraints = ob.OBFFConstraints()

    # move building unit to the origin and align on z axis
    unit_inp = trans_origin(unit_inp, atom1_inp)
    unit_inp = alignZ(unit_inp, atom1_inp, atom2_inp)

    obConversion.SetInAndOutFormats("xyz", "xyz")
    gen_xyz(xyz_tmp_dir + unit_name + '_mol.xyz', unit_inp)
    mol = OBmolUpdateXYZcoordinates(ref_xyz, xyz_tmp_dir + unit_name + '_mol.xyz')

    dis = 0.25

    count = 0
    list_energy = []
    while count >= 0:  # Infinite loop
        a = mol.GetAtom(atom1_inp + 1)
        a.SetVector(
            unit_inp.loc[atom1_inp, 1],
            unit_inp.loc[atom1_inp, 2],
            unit_inp.loc[atom1_inp, 3] - count * dis,
        )

        b = mol.GetAtom(dum1_inp + 1)
        b.SetVector(
            unit_inp.loc[dum1_inp, 1],
            unit_inp.loc[dum1_inp, 2],
            unit_inp.loc[dum1_inp, 3] - count * dis,
        )

        c = mol.GetAtom(atom2_inp + 1)
        c.SetVector(
            unit_inp.loc[atom2_inp, 1],
            unit_inp.loc[atom2_inp, 2],
            unit_inp.loc[atom2_inp, 3] + count * dis,
        )

        d = mol.GetAtom(dum2_inp + 1)
        d.SetVector(
            unit_inp.loc[dum2_inp, 1],
            unit_inp.loc[dum2_inp, 2],
            unit_inp.loc[dum2_inp, 3] + count * dis,
        )

        for atom_id in [dum1_inp + 1, dum2_inp + 1, atom1_inp + 1, atom2_inp + 1]:
            constraints.AddAtomConstraint(atom_id)

        ff.Setup(mol, constraints)
        ff.SteepestDescent(1000)

        ff.UpdateCoordinates(mol)

        if count == 0:
            slope = 0
        else:
            slope = (ff.Energy() - list_energy[-1][1]) / (count - list_energy[-1][0])

        list_energy.append([count, ff.Energy(), slope])
        obConversion.WriteFile(
            mol, xyz_tmp_dir + unit_name + '_output' + str(count) + '.xyz'
        )

        if (
            CompareConnectInfo(
                xyz_tmp_dir + unit_name + '_mol.xyz',
                xyz_tmp_dir + unit_name + '_output' + str(count) + '.xyz',
            )
            == 'WRONG'
        ):

            return (
                pd.read_csv(
                    xyz_tmp_dir + unit_name + '_mol.xyz',
                    header=None,
                    skiprows=2,
                    delim_whitespace=True,
                ),
                xyz_tmp_dir + unit_name + '_mol.xyz',
            )

        elif slope > Tol_ChainCorr and count <= 3:

            return (
                pd.read_csv(
                    xyz_tmp_dir + unit_name + '_mol.xyz',
                    header=None,
                    skiprows=2,
                    delim_whitespace=True,
                ),
                xyz_tmp_dir + unit_name + '_mol.xyz',
            )
        elif slope > Tol_ChainCorr and count > 3:

            # optimize molecule again with large number of steps
            ff.SteepestDescent(1000000, 0.0000001)
            ff.UpdateCoordinates(mol)
            obConversion.WriteFile(
                mol, xyz_tmp_dir + unit_name + '_final' + str(count) + '.xyz'
            )

            return (
                pd.read_csv(
                    xyz_tmp_dir + unit_name + '_final' + str(count) + '.xyz',
                    header=None,
                    skiprows=2,
                    delim_whitespace=True,
                ),
                xyz_tmp_dir + unit_name + '_final' + str(count) + '.xyz',
            )

        else:
            count += 1


def Init_info(unit_name, smiles_each_ori, xyz_in_dir, length):
    # Get index of dummy atoms and bond type associated with it
    try:
        dum_index, bond_type = FetchDum(smiles_each_ori)
        if len(dum_index) == 2:
            dum1 = dum_index[0]
            dum2 = dum_index[1]
        else:
            print(
                unit_name,
                ": There are more or less than two dummy atoms in the SMILES string; "
                "Hint: PSP works only for one-dimensional polymers.",
            )
            return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'
    except Exception:
        print(
            unit_name,
            ": Couldn't fetch the position of dummy atoms. Hints: (1) In SMILES strings, use '*' for a dummy atom,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    # Assign dummy atom according to bond type
    if bond_type == 'SINGLE':
        dum, unit_dis = 'Cl', -0.17
        # List of oligomers
        oligo_list = list(set(length) - set(['n']))
    elif bond_type == 'DOUBLE':
        dum, unit_dis = 'O', 0.25
        # List of oligomers
        oligo_list = []
    else:
        print(
            unit_name,
            ": Unusal bond type (Only single or double bonds are acceptable)."
            "Hints: (1) Check bonds between the dummy and connecting atoms in SMILES string"
            "       (2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    # Replace '*' with dummy atom
    smiles_each = smiles_each_ori.replace(r'*', dum)

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz, m1 = smiles_xyz(unit_name, smiles_each, xyz_in_dir)

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(
            unit_name,
            ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    # Collect valency and connecting information for each atom
    neigh_atoms_info = connec_info(xyz_in_dir + unit_name + '.xyz')

    try:
        # Find connecting atoms associated with dummy atoms.
        # dum1 and dum2 are connected to atom1 and atom2, respectively.
        atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]
        atom2 = neigh_atoms_info['NeiAtom'][dum2].copy()[0]

    except Exception:
        print(
            unit_name,
            ": Couldn't get the position of connecting atoms. Hints: (1) XYZ coordinates are not acceptable,"
            "(2) Check Open Babel installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'
    return (
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
        '',
    )


# Build a polymer from a monomer unit. (main function)
# 1. Rotate all single bonds and find out the best monomer by maximizing angle between two possible vectors between two
# sets of connecting and dummy atoms.
# 2. Distance between two dummy atoms is considered as the second criteria to choose a good monomer unit.
# 3. If a good monomer is found, then build a dimer.
# 4. If a good monomer is not found, combine two monomers (flip/rotate one unit) and consider it as a new monomer unit
# 5. Search the best unit and build a dimer
# 6. Always check connectivity between atoms to verify if a monomer/dimer is not acceptable or not.
# 7. Minimize final geometry using Steepest Descent


def build_polymer(
    unit_name,
    df_smiles,
    ID,
    SMILES,
    xyz_in_dir,
    xyz_tmp_dir,
    vasp_out_dir,
    rot_angles_monomer,
    rot_angles_dimer,
    Steps,
    Substeps,
    num_conf,
    length,
    method,
    IntraChainCorr,
    Tol_ChainCorr,
    Inter_Chain_Dis,
):
    print(" Chain model building started for", unit_name, "...")
    vasp_out_dir_indi = vasp_out_dir + unit_name + '/'
    build_dir(vasp_out_dir_indi)

    # Initial values
    decision = 'FAILURE'
    SN = 0

    # Get SMILES
    smiles_each = df_smiles[df_smiles[ID] == unit_name][SMILES].values[0]

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
    ) = Init_info(unit_name, smiles_each, xyz_in_dir, length)

    # keep a copy of idx of dummy and linking atoms
    # These idx will be used if there is no rotatable bonds
    dum1_smi, dum2_smi, atom1_smi, atom2_smi = dum1, dum2, atom1, atom2

    if flag == 'REJECT':
        return unit_name, 'REJECT', 0

    if atom1 == atom2:
        smiles_each = gen_dimer_smiles(dum1, dum2, atom1, atom2, smiles_each)

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
        ) = Init_info(unit_name, smiles_each, xyz_in_dir, length)

        if flag == 'REJECT':
            return unit_name, 'REJECT', 0

    # create 100 conformers and select which has the largest dihedral angle (within 8 degree) and lowest energy
    find_best_conf(unit_name, m1, dum1, dum2, atom1, atom2, xyz_in_dir)

    # Minimize geometry using steepest descent
    unit = localopt(
        unit_name,
        xyz_in_dir + unit_name + '.xyz',
        dum1,
        dum2,
        atom1,
        atom2,
        xyz_tmp_dir,
    )

    # Rearrange rows
    rows = unit.index.tolist()
    for i in [dum1, atom1, atom2, dum2]:
        rows.remove(i)
    new_rows = [dum1, atom1, atom2, dum2] + rows
    unit = unit.loc[new_rows].reset_index(drop=True)
    dum1, atom1, atom2, dum2 = 0, 1, 2, 3

    gen_xyz(xyz_tmp_dir + unit_name + '_rearranged.xyz', unit)

    # update neigh_atoms_info
    neigh_atoms_info = connec_info(xyz_tmp_dir + unit_name + '_rearranged.xyz')

    # Stretch the repeating unit
    if IntraChainCorr == 1:
        unit, unit_init_xyz = MakePolymerStraight(
            unit_name,
            xyz_tmp_dir + unit_name + '_rearranged.xyz',
            unit,
            dum1,
            dum2,
            atom1,
            atom2,
            xyz_tmp_dir,
            Tol_ChainCorr,
        )

    check_connectivity_dimer = mono2dimer(
        unit_name, unit, 'CORRECT', dum1, dum2, atom1, atom2, unit_dis
    )

    # building unit is copied and may be used in building flexible dimers
    unit_copied = unit.copy()

    if check_connectivity_dimer == 'CORRECT':
        decision = 'SUCCESS'
        SN += 1

        if 'n' in length:
            gen_vasp(
                vasp_out_dir_indi,
                unit_name,
                unit,
                dum1,
                dum2,
                atom1,
                atom2,
                dum,
                unit_dis,
                Inter_Chain_Dis=Inter_Chain_Dis,
                Polymer=True,
            )

        if len(oligo_list) > 0:
            for oligo_len in oligo_list:
                (
                    oligomer,
                    dum1_oligo,
                    atom1_oligo,
                    dum2_oligo,
                    atom2_oligo,
                ) = oligomer_build(
                    unit,
                    unit_name,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    oligo_len,
                    unit_dis,
                    neigh_atoms_info,
                )
                gen_vasp(
                    vasp_out_dir_indi,
                    unit_name,
                    oligomer,
                    dum1_oligo,
                    dum2_oligo,
                    atom1_oligo,
                    atom2_oligo,
                    dum,
                    unit_dis,
                    length=oligo_len,
                    Inter_Chain_Dis=Inter_Chain_Dis,
                )

                # Remove dummy atoms
                oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                if SN > 1:
                    xyz_file_name = (
                        vasp_out_dir_indi
                        + unit_name
                        + '_'
                        + str(SN)
                        + '_N'
                        + str(oligo_len)
                        + '.xyz'
                    )
                else:
                    xyz_file_name = (
                        vasp_out_dir_indi + unit_name + '_N' + str(oligo_len) + '.xyz'
                    )

                gen_xyz(xyz_file_name, oligomer)

        if SN >= num_conf:
            print(" Chain model building completed for", unit_name, ".")
            return unit_name, 'SUCCESS', SN

    # Simulated Annealing
    if method == 'SA' and SN < num_conf:
        print(" Entering simulated annealing steps", unit_name, "...")
        #  Find single bonds and rotate
        single_bond = single_bonds(unit_name, unit, xyz_tmp_dir)

        isempty = single_bond.empty
        if isempty is True and SN < num_conf:
            print(unit_name, "No rotatable single bonds, building a dimer ...")

            smiles_each = gen_dimer_smiles(
                dum1_smi, dum2_smi, atom1_smi, atom2_smi, smiles_each
            )
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
            ) = Init_info(unit_name, smiles_each, xyz_in_dir, length)

            if flag == 'REJECT':
                return unit_name, 'REJECT', 0

            # create 100 conformers and select which has the largest dihedral angle (within 8 degree) and lowest energy
            find_best_conf(unit_name, m1, dum1, dum2, atom1, atom2, xyz_in_dir)

            # Minimize geometry using steepest descent
            unit = localopt(
                unit_name,
                xyz_in_dir + unit_name + '.xyz',
                dum1,
                dum2,
                atom1,
                atom2,
                xyz_tmp_dir,
            )

            # Rearrange rows
            rows = unit.index.tolist()
            for i in [dum1, atom1, atom2, dum2]:
                rows.remove(i)
            new_rows = [dum1, atom1, atom2, dum2] + rows
            unit = unit.loc[new_rows].reset_index(drop=True)
            dum1, atom1, atom2, dum2 = 0, 1, 2, 3

            gen_xyz(xyz_tmp_dir + unit_name + '_rearranged.xyz', unit)

            # update neigh_atoms_info
            neigh_atoms_info = connec_info(xyz_tmp_dir + unit_name + '_rearranged.xyz')

            # Stretch the repeating unit
            if IntraChainCorr == 1:
                unit, unit_init_xyz = MakePolymerStraight(
                    unit_name,
                    xyz_tmp_dir + unit_name + '_rearranged.xyz',
                    unit,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    xyz_tmp_dir,
                    Tol_ChainCorr,
                )

            check_connectivity_dimer = mono2dimer(
                unit_name, unit, 'CORRECT', dum1, dum2, atom1, atom2, unit_dis
            )

            # building unit is copied and may be used in building flexible dimers
            unit_copied = unit.copy()

            if check_connectivity_dimer == 'CORRECT':
                decision = 'SUCCESS'
                SN += 1

                if 'n' in length:
                    gen_vasp(
                        vasp_out_dir_indi,
                        unit_name,
                        unit,
                        dum1,
                        dum2,
                        atom1,
                        atom2,
                        dum,
                        unit_dis,
                        Inter_Chain_Dis=Inter_Chain_Dis,
                        Polymer=True,
                    )

                if len(oligo_list) > 0:
                    for oligo_len in oligo_list:
                        (
                            oligomer,
                            dum1_oligo,
                            atom1_oligo,
                            dum2_oligo,
                            atom2_oligo,
                        ) = oligomer_build(
                            unit,
                            unit_name,
                            dum1,
                            dum2,
                            atom1,
                            atom2,
                            oligo_len,
                            unit_dis,
                            neigh_atoms_info,
                        )
                        gen_vasp(
                            vasp_out_dir_indi,
                            unit_name,
                            oligomer,
                            dum1_oligo,
                            dum2_oligo,
                            atom1_oligo,
                            atom2_oligo,
                            dum,
                            unit_dis,
                            length=oligo_len,
                            Inter_Chain_Dis=Inter_Chain_Dis,
                        )

                        # Remove dummy atoms
                        oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                        if SN > 1:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_'
                                + str(SN)
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )
                        else:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )

                        gen_xyz(xyz_file_name, oligomer)

                if SN >= num_conf:
                    print(" Chain model building completed for", unit_name, ".")
                    return unit_name, 'SUCCESS', SN
                else:
                    #  Find single bonds and rotate
                    single_bond = single_bonds(unit_name, unit, xyz_tmp_dir)

        if isempty is True and SN > 0:
            print(" Chain model building completed for", unit_name, ".")
            return unit_name, 'SUCCESS', SN
        if isempty is True and SN == 0:
            return unit_name, 'FAILURE', 0

        results = an.SA(
            unit_name,
            unit,
            single_bond,
            rot_angles_monomer,
            neigh_atoms_info,
            xyz_tmp_dir,
            dum1,
            dum2,
            atom1,
            atom2,
            Steps,
            Substeps,
        )

        results = results.sort_index(ascending=False)

        # Keep num_conf+10 rows to reduce computational costs
        results = results.head(num_conf + 10)
        TotalSN, first_conf_saved = 0, 0
        for index, row in results.iterrows():
            TotalSN += 1

            # Minimize geometry using steepest descent
            final_unit = localopt(
                unit_name, row['xyzFile'], dum1, dum2, atom1, atom2, xyz_tmp_dir
            )

            # Stretch the repeating unit
            if IntraChainCorr == 1:
                final_unit, final_unit_xyz = MakePolymerStraight(
                    unit_name,
                    xyz_tmp_dir + unit_name + '_rearranged.xyz',
                    final_unit,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    xyz_tmp_dir,
                    Tol_ChainCorr,
                )
            else:
                final_unit_xyz = row['xyzFile']

            # Check connectivity
            check_connectivity_monomer = 'CORRECT'

            neigh_atoms_info_new = connec_info(final_unit_xyz)
            for row in neigh_atoms_info.index.tolist():
                if sorted(neigh_atoms_info.loc[row]['NeiAtom']) != sorted(
                    neigh_atoms_info_new.loc[row]['NeiAtom']
                ):
                    check_connectivity_monomer = 'WRONG'

            if check_connectivity_monomer == 'CORRECT' and first_conf_saved == 0:
                # building unit is copied and may be used in building flexible dimers
                unit_copied = final_unit.copy()
                first_conf_saved = 1

            check_connectivity_dimer = mono2dimer(
                unit_name,
                final_unit,
                check_connectivity_monomer,
                dum1,
                dum2,
                atom1,
                atom2,
                unit_dis,
            )

            if check_connectivity_dimer == 'CORRECT':
                decision = 'SUCCESS'
                SN += 1
                if 'n' in length:
                    gen_vasp(
                        vasp_out_dir_indi,
                        unit_name,
                        final_unit,
                        dum1,
                        dum2,
                        atom1,
                        atom2,
                        dum,
                        unit_dis,
                        Inter_Chain_Dis=Inter_Chain_Dis,
                        Polymer=True,
                    )

                if len(oligo_list) > 0:
                    for oligo_len in oligo_list:
                        (
                            oligomer,
                            dum1_oligo,
                            atom1_oligo,
                            dum2_oligo,
                            atom2_oligo,
                        ) = oligomer_build(
                            final_unit,
                            unit_name,
                            dum1,
                            dum2,
                            atom1,
                            atom2,
                            oligo_len,
                            unit_dis,
                            neigh_atoms_info_new,
                        )
                        gen_vasp(
                            vasp_out_dir_indi,
                            unit_name,
                            oligomer,
                            dum1_oligo,
                            dum2_oligo,
                            atom1_oligo,
                            atom2_oligo,
                            dum,
                            unit_dis,
                            length=oligo_len,
                            Inter_Chain_Dis=Inter_Chain_Dis,
                        )

                        # Remove dummy atoms
                        oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                        if SN > 1:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_'
                                + str(SN)
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )
                        else:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )

                        gen_xyz(xyz_file_name, oligomer)

                if SN == num_conf:
                    break

            # If we do not get a proper monomer unit, then consider a dimer as a monomer unit and
            # build a dimer of the same
            if SN < num_conf and TotalSN == results.index.size:
                unit = unit_copied.copy()
                unit = trans_origin(unit, atom1)
                unit = alignZ(unit, atom1, atom2)
                (
                    unit_dimer,
                    neigh_atoms_info_dimer,
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                ) = build_dimer_rotate(
                    unit_name,
                    rot_angles_dimer,
                    unit,
                    unit,
                    dum,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    unit_dis,
                )
                isempty = unit_dimer.empty

                if isempty is True and SN == 0:
                    print(unit_name, "Couldn't find an acceptable dimer.")
                    return unit_name, 'FAILURE', 0
                elif isempty is True and SN > 0:
                    print(" Chain model building completed for", unit_name, ".")
                    return unit_name, 'SUCCESS', SN

                # Generate XYZ file
                gen_xyz(xyz_tmp_dir + unit_name + '_dimer.xyz', unit_dimer)

                # Minimize geometry using steepest descent
                unit_dimer = localopt(
                    unit_name,
                    xyz_tmp_dir + unit_name + '_dimer.xyz',
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                    xyz_tmp_dir,
                )

                # Stretch the repeating unit
                if IntraChainCorr == 1:
                    unit_dimer, unit_dimer_xyz = MakePolymerStraight(
                        unit_name,
                        xyz_tmp_dir + unit_name + '_dimer.xyz',
                        unit_dimer,
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        xyz_tmp_dir,
                        Tol_ChainCorr,
                    )

                check_connectivity_dimer = mono2dimer(
                    unit_name,
                    unit_dimer,
                    'CORRECT',
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                    unit_dis,
                )

                if check_connectivity_dimer == 'CORRECT':
                    decision = 'SUCCESS'
                    SN += 1
                    if 'n' in length:
                        gen_vasp(
                            vasp_out_dir_indi,
                            unit_name,
                            unit_dimer,
                            dum1_2nd,
                            dum2_2nd,
                            atom1_2nd,
                            atom2_2nd,
                            dum,
                            unit_dis,
                            Inter_Chain_Dis=Inter_Chain_Dis,
                            Polymer=True,
                        )

                    if len(oligo_list) > 0:
                        for oligo_len in oligo_list:
                            (
                                oligomer,
                                dum1_oligo,
                                atom1_oligo,
                                dum2_oligo,
                                atom2_oligo,
                            ) = oligomer_build(
                                unit_dimer,
                                unit_name,
                                dum1_2nd,
                                dum2_2nd,
                                atom1_2nd,
                                atom2_2nd,
                                oligo_len,
                                unit_dis,
                                neigh_atoms_info_dimer,
                            )
                            gen_vasp(
                                vasp_out_dir_indi,
                                unit_name,
                                oligomer,
                                dum1_oligo,
                                dum2_oligo,
                                atom1_oligo,
                                atom2_oligo,
                                dum,
                                unit_dis,
                                length=oligo_len,
                                Inter_Chain_Dis=Inter_Chain_Dis,
                            )

                            # Remove dummy atoms
                            oligomer = oligomer.drop([dum1_oligo, dum2_oligo])
                            if SN > 1:
                                xyz_file_name = (
                                    vasp_out_dir_indi
                                    + unit_name
                                    + '_'
                                    + str(SN)
                                    + '_N'
                                    + str(oligo_len)
                                    + '.xyz'
                                )
                            else:
                                xyz_file_name = (
                                    vasp_out_dir_indi
                                    + unit_name
                                    + '_N'
                                    + str(oligo_len)
                                    + '.xyz'
                                )

                            gen_xyz(xyz_file_name, oligomer)

                    if SN == num_conf:
                        break

                # Generate XYZ file and find connectivity
                # gen_xyz(xyz_tmp_dir + unit_name + '_dimer.xyz', unit_dimer)
                neigh_atoms_info_dimer = connec_info(
                    xyz_tmp_dir + unit_name + '_dimer.xyz'
                )

                #  Find single bonds and rotate
                single_bond_dimer = single_bonds(unit_name, unit_dimer, xyz_tmp_dir)

                isempty = single_bond_dimer.empty
                if isempty is True and SN == 0:
                    print(unit_name, "No rotatable single bonds in dimer")
                    return unit_name, 'FAILURE', 0
                elif isempty is True and SN > 0:
                    print(" Chain model building completed for", unit_name, ".")
                    return unit_name, 'SUCCESS', SN

                results = an.SA(
                    unit_name,
                    unit_dimer,
                    single_bond_dimer,
                    rot_angles_monomer,
                    neigh_atoms_info_dimer,
                    xyz_tmp_dir,
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                    Steps,
                    Substeps,
                )
                results = results.sort_index(ascending=False)

                for index, row in results.iterrows():

                    # Minimize geometry using steepest descent
                    final_unit = localopt(
                        unit_name,
                        row['xyzFile'],
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        xyz_tmp_dir,
                    )

                    # Stretch the repeating unit
                    if IntraChainCorr == 1:
                        final_unit, final_unit_xyz = MakePolymerStraight(
                            unit_name,
                            xyz_tmp_dir + unit_name + '_dimer.xyz',
                            final_unit,
                            dum1_2nd,
                            dum2_2nd,
                            atom1_2nd,
                            atom2_2nd,
                            xyz_tmp_dir,
                            Tol_ChainCorr,
                        )
                    else:
                        final_unit_xyz = row['xyzFile']

                    # Check Connectivity
                    check_connectivity_monomer = 'CORRECT'

                    neigh_atoms_info_new = connec_info(final_unit_xyz)
                    for row in neigh_atoms_info_dimer.index.tolist():
                        if sorted(neigh_atoms_info_dimer.loc[row]['NeiAtom']) != sorted(
                            neigh_atoms_info_new.loc[row]['NeiAtom']
                        ):
                            check_connectivity_monomer = 'WRONG'

                    check_connectivity_dimer = mono2dimer(
                        unit_name,
                        final_unit,
                        check_connectivity_monomer,
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        unit_dis,
                    )
                    if check_connectivity_dimer == 'CORRECT':
                        decision = 'SUCCESS'
                        SN += 1
                        if 'n' in length:
                            gen_vasp(
                                vasp_out_dir_indi,
                                unit_name,
                                final_unit,
                                dum1_2nd,
                                dum2_2nd,
                                atom1_2nd,
                                atom2_2nd,
                                dum,
                                unit_dis,
                                Inter_Chain_Dis=Inter_Chain_Dis,
                                Polymer=True,
                            )

                        if len(oligo_list) > 0:
                            for oligo_len in oligo_list:
                                (
                                    oligomer,
                                    dum1_oligo,
                                    atom1_oligo,
                                    dum2_oligo,
                                    atom2_oligo,
                                ) = oligomer_build(
                                    final_unit,
                                    unit_name,
                                    dum1_2nd,
                                    dum2_2nd,
                                    atom1_2nd,
                                    atom2_2nd,
                                    oligo_len,
                                    unit_dis,
                                    neigh_atoms_info_new,
                                )
                                gen_vasp(
                                    vasp_out_dir_indi,
                                    unit_name,
                                    oligomer,
                                    dum1_oligo,
                                    dum2_oligo,
                                    atom1_oligo,
                                    atom2_oligo,
                                    dum,
                                    unit_dis,
                                    length=oligo_len,
                                    Inter_Chain_Dis=Inter_Chain_Dis,
                                )

                                # Remove dummy atoms
                                oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                                if SN > 1:
                                    xyz_file_name = (
                                        vasp_out_dir_indi
                                        + unit_name
                                        + '_'
                                        + str(SN)
                                        + '_N'
                                        + str(oligo_len)
                                        + '.xyz'
                                    )
                                else:
                                    xyz_file_name = (
                                        vasp_out_dir_indi
                                        + unit_name
                                        + '_N'
                                        + str(oligo_len)
                                        + '.xyz'
                                    )

                                gen_xyz(xyz_file_name, oligomer)

                        if SN == num_conf:
                            break

    elif method == 'Dimer' and SN < num_conf:
        print(" Generating dimers", unit_name, "...")
        SN = 0
        for angle in rot_angles_dimer:
            unit1 = unit.copy()
            unit2 = unit.copy()
            unit2 = trans_origin(unit2, atom1)
            unit2 = alignZ(unit2, atom1, atom2)
            unit2 = rotateZ(unit2, angle, np.arange(len(unit2[0].values)))

            (
                dimer,
                check_connectivity_monomer,
                dum1_2nd,
                dum2_2nd,
                atom1_2nd,
                atom2_2nd,
            ) = TwoMonomers_Dimer(
                unit_name, unit1, unit2, dum1, dum2, atom1, atom2, dum, unit_dis
            )
            if atom1_2nd != atom2_2nd:
                if check_connectivity_monomer == 'CORRECT':
                    unit = dimer.copy()
                    # Build a dimer
                    unit = trans_origin(unit, dum1_2nd)
                    unit = alignZ(unit, dum1_2nd, dum2_2nd)
                    polymer, check_connectivity_dimer = build(
                        unit_name,
                        2,
                        unit,
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        unit_dis,
                    )

                    if check_connectivity_dimer == 'CORRECT':
                        decision = 'SUCCESS'
                        SN += 1
                        if 'n' in length:
                            gen_vasp(
                                vasp_out_dir_indi,
                                unit_name,
                                dimer,
                                dum1_2nd,
                                dum2_2nd,
                                atom1_2nd,
                                atom2_2nd,
                                dum,
                                unit_dis,
                                Inter_Chain_Dis=Inter_Chain_Dis,
                                Polymer=True,
                            )

                        if SN == num_conf:
                            break
    print(" Chain model building completed for", unit_name, ".")
    return unit_name, decision, SN


def build_3D(
    unit_name,
    df_smiles,
    ID,
    SMILES,
    LeftCap,
    RightCap,
    out_dir,
    Inter_Mol_Dis,
    Length,
    xyz_in_dir,
    NumConf,
    loop,
    IrrStruc,
    OPLS,
    GAFF2,
    atom_typing_,
    NCores_opt,
):
    LCap_ = False
    if LeftCap in df_smiles.columns:
        smiles_LCap_ = df_smiles[df_smiles[ID] == unit_name][LeftCap].values[0]
        if is_nan(smiles_LCap_) is False:
            LCap_ = True

    else:
        smiles_LCap_ = ''

    RCap_ = False
    if RightCap in df_smiles.columns:
        smiles_RCap_ = df_smiles[df_smiles[ID] == unit_name][RightCap].values[0]
        if is_nan(smiles_RCap_) is False:
            RCap_ = True

    else:
        smiles_RCap_ = ''

    # Get SMILES
    smiles_each = df_smiles[df_smiles[ID] == unit_name][SMILES].values[0]
    # smiles_each_copy = copy.copy(smiles_each)

    # count = 0
    Final_SMILES = []
    for ln in Length:
        # start_1 = time.time()
        if ln == 1:
            if LCap_ is False and RCap_ is False:
                mol = Chem.MolFromSmiles(smiles_each)
                mol_new = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#0]'))
                smiles_each_ind = Chem.MolToSmiles(mol_new)
            else:
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
                ) = Init_info(unit_name, smiles_each, xyz_in_dir, Length)

                if flag == 'REJECT' and len(Final_SMILES) == 0 and ln == Length[-1]:
                    return unit_name, 'REJECT', Final_SMILES
                elif flag == 'REJECT' and len(Final_SMILES) >= 1 and ln == Length[-1]:
                    return unit_name, 'PARTIAL SUCCESS', Final_SMILES
                # Join end caps
                smiles_each_ind = gen_smiles_with_cap(
                    unit_name,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    smiles_each,
                    smiles_LCap_,
                    smiles_RCap_,
                    LCap_,
                    RCap_,
                    xyz_in_dir,
                )

        elif ln > 1:
            # smiles_each = copy.copy(smiles_each_copy)

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
            ) = Init_info(unit_name, smiles_each, xyz_in_dir, Length)

            if flag == 'REJECT' and len(Final_SMILES) == 0 and ln == Length[-1]:
                return unit_name, 'REJECT', Final_SMILES
            elif flag == 'REJECT' and len(Final_SMILES) >= 1 and ln == Length[-1]:
                return unit_name, 'PARTIAL SUCCESS', Final_SMILES

            smiles_each_ind = gen_oligomer_smiles(
                unit_name,
                dum1,
                dum2,
                atom1,
                atom2,
                smiles_each,
                ln,
                loop,
                smiles_LCap_,
                LCap_,
                smiles_RCap_,
                RCap_,
                xyz_in_dir,
            )

        m1 = Chem.MolFromSmiles(smiles_each_ind)
        if m1 is None and len(Final_SMILES) == 0 and ln == Length[-1]:
            return unit_name, 'REJECT', Final_SMILES
        elif m1 is None and len(Final_SMILES) >= 1 and ln == Length[-1]:
            return unit_name, 'PARTIAL SUCCESS', Final_SMILES

        Final_SMILES.append(smiles_each_ind)
        # OB_smi_2_xyz_vasp(unit_name, smiles_each_ind, l, out_dir, Inter_Mol_Dis, NumConf=NumConf, seed=None)
        NumC = gen_conf_xyz_vasp(
            unit_name,
            m1,
            out_dir,
            ln,
            NumConf,
            Inter_Mol_Dis,
            IrrStruc,
            NCores_opt,
            OPLS,
            GAFF2,
            atom_typing_,
        )

        if NumC == 0 and ln == Length[-1]:
            return unit_name, 'FAILURE', Final_SMILES
        elif ln == Length[-1]:
            return unit_name, 'SUCCESS', Final_SMILES

        # end_1 = time.time()
        # print(l, end_1 - start_1)
    # return unit_name, 'SUCCESS', Final_SMILES


def Init_info_Cap(unit_name, smiles_each_ori, xyz_in_dir):
    # Get index of dummy atoms and bond type associated with it
    try:
        dum_index, bond_type = FetchDum(smiles_each_ori)
        if len(dum_index) == 1:
            dum1 = dum_index[0]
        else:
            print(
                unit_name,
                ": There are more or less than one dummy atoms in the SMILES string; ",
            )
            return unit_name, 0, 0, 0, 0, 'REJECT'
    except Exception:
        print(
            unit_name,
            ": Couldn't fetch the position of dummy atoms. Hints: (1) In SMILES string, use '*' for a dummy atom,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 'REJECT'

    # Replace '*' with dummy atom
    smiles_each = smiles_each_ori.replace(r'*', 'Cl')

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz, m1 = smiles_xyz(unit_name, smiles_each, xyz_in_dir)

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(
            unit_name,
            ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 'REJECT'

    # Collect valency and connecting information for each atom
    neigh_atoms_info = connec_info(xyz_in_dir + unit_name + '.xyz')

    try:
        # Find connecting atoms associated with dummy atoms.
        # dum1 and dum2 are connected to atom1 and atom2, respectively.
        atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]

    except Exception:
        print(
            unit_name,
            ": Couldn't get the position of connecting atoms. Hints: (1) XYZ coordinates are not acceptable,"
            "(2) Check Open Babel installation.",
        )
        return unit_name, 0, 0, 0, 0, 'REJECT'
    return (
        unit_name,
        dum1,
        atom1,
        m1,
        neigh_atoms_info,
        '',
    )


def gen_smiles_with_cap(
    unit_name,
    dum1,
    dum2,
    atom1,
    atom2,
    smiles_each,
    smiles_LCap_,
    smiles_RCap_,
    LCap_,
    RCap_,
    xyz_in_dir,
    WithDum=True,
):
    # Main chain
    # Check if there are dummy atoms in the chain
    if WithDum is True:
        main_mol = Chem.MolFromSmiles(smiles_each)
        main_edit_m1 = Chem.EditableMol(main_mol)

        # Remove dummy atoms
        main_edit_m1.RemoveAtom(dum1)
        if dum1 < dum2:
            main_edit_m1.RemoveAtom(dum2 - 1)
        else:
            main_edit_m1.RemoveAtom(dum2)

        # Mol without dummy atom
        main_mol_noDum = main_edit_m1.GetMol()

        # Get linking atoms
        if atom1 > atom2:
            atom1, atom2 = atom2, atom1

        if dum1 < atom1 and dum2 < atom1:
            first_atom = atom1 - 2
        elif (dum1 < atom1 and dum2 > atom1) or (dum1 > atom1 and dum2 < atom1):
            first_atom = atom1 - 1
        else:
            first_atom = atom1

        if dum1 < atom2 and dum2 < atom2:
            second_atom = atom2 - 2
        elif (dum1 < atom2 and dum2 > atom2) or (dum1 > atom2 and dum2 < atom2):
            second_atom = atom2 - 1
        else:
            second_atom = atom2
    else:
        main_mol_noDum = smiles_each
        first_atom, second_atom = atom1, atom2

    LCap_add = 0
    # Left Cap
    if LCap_ is True:
        (unit_name, dum_L, atom_L, m1L, neigh_atoms_info_L, flag_L) = Init_info_Cap(
            unit_name, smiles_LCap_, xyz_in_dir
        )

        # Reject if SMILES is not correct
        if flag_L == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for LeftCap
        LCap_m1 = Chem.MolFromSmiles(smiles_LCap_)
        LCap_edit_m1 = Chem.EditableMol(LCap_m1)

        # Remove dummy atoms
        LCap_edit_m1.RemoveAtom(dum_L)

        # Mol without dummy atom
        LCap_m1 = LCap_edit_m1.GetMol()
        LCap_add = LCap_m1.GetNumAtoms()

        # Linking atom
        if dum_L < atom_L:
            LCap_atom = atom_L - 1
        else:
            LCap_atom = atom_L

        # Join main chain with Left Cap
        combo = Chem.CombineMols(LCap_m1, main_mol_noDum)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            LCap_atom, first_atom + LCap_add, order=Chem.rdchem.BondType.SINGLE
        )
        main_mol_noDum = edcombo.GetMol()

    # Right Cap
    if RCap_ is True:
        (unit_name, dum_R, atom_R, m1L, neigh_atoms_info_R, flag_R) = Init_info_Cap(
            unit_name, smiles_RCap_, xyz_in_dir
        )

        # Reject if SMILES is not correct
        if flag_R == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for RightCap
        RCap_m1 = Chem.MolFromSmiles(smiles_RCap_)
        RCap_edit_m1 = Chem.EditableMol(RCap_m1)

        # Remove dummy atoms
        RCap_edit_m1.RemoveAtom(dum_R)

        # Mol without dummy atom
        RCap_m1 = RCap_edit_m1.GetMol()

        # Linking atom
        if dum_R < atom_R:
            RCap_atom = atom_R - 1
        else:
            RCap_atom = atom_R

        # Join main chain with Left Cap
        combo = Chem.CombineMols(main_mol_noDum, RCap_m1)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            LCap_add + second_atom,
            RCap_atom + main_mol_noDum.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE,
        )
        main_mol_noDum = edcombo.GetMol()
    return Chem.MolToSmiles(main_mol_noDum)


def gen_oligomer_smiles(
    unit_name,
    dum1,
    dum2,
    atom1,
    atom2,
    input_smiles,
    ln,
    loop,
    smiles_LCap_,
    LCap_,
    smiles_RCap_,
    RCap_,
    xyz_in_dir,
):
    input_mol = Chem.MolFromSmiles(input_smiles)
    edit_m1 = Chem.EditableMol(input_mol)

    edit_m1.RemoveAtom(dum1)

    if dum1 < dum2:
        edit_m1.RemoveAtom(dum2 - 1)
    else:
        edit_m1.RemoveAtom(dum2)

    monomer_mol = edit_m1.GetMol()
    inti_mol = monomer_mol

    if atom1 > atom2:
        atom1, atom2 = atom2, atom1

    if dum1 < atom1 and dum2 < atom1:
        second_atom = atom1 - 2
    elif (dum1 < atom1 and dum2 > atom1) or (dum1 > atom1 and dum2 < atom1):
        second_atom = atom1 - 1
    else:
        second_atom = atom1

    if dum1 < atom2 and dum2 < atom2:
        first_atom = atom2 - 2
    elif (dum1 < atom2 and dum2 > atom2) or (dum1 > atom2 and dum2 < atom2):
        first_atom = atom2 - 1
    else:
        first_atom = atom2

    for i in range(1, ln):
        combo = Chem.CombineMols(inti_mol, monomer_mol)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            second_atom + (i - 1) * monomer_mol.GetNumAtoms(),
            first_atom + i * monomer_mol.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE,
        )
        inti_mol = edcombo.GetMol()

    if loop is True and LCap_ is False and RCap_ is False:
        edcombo.AddBond(
            first_atom,
            second_atom + i * monomer_mol.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE,
        )
        inti_mol = edcombo.GetMol()

    if LCap_ is True or RCap_ is True:
        inti_mol = gen_smiles_with_cap(
            unit_name,
            0,
            0,
            first_atom,
            second_atom + i * monomer_mol.GetNumAtoms(),
            inti_mol,
            smiles_LCap_,
            smiles_RCap_,
            LCap_,
            RCap_,
            xyz_in_dir,
            WithDum=False,
        )

        return inti_mol

    return Chem.MolToSmiles(inti_mol)


def OB_smi_2_xyz_vasp(
    unit_name, smiles, length, out_dir, Inter_Mol_Dis, NumConf=1, seed=None,
):
    if seed is not None:
        rand = ob.OBRandom(True)
        rand.Seed(seed)

    obConversion.SetInFormat("smi")

    mol = ob.OBMol()
    obConversion.ReadString(mol, smiles)
    mol.AddHydrogens()

    # Reduce opt steps for large molecules
    if mol.NumAtoms() > 50000:
        OptStepCG1 = 100
        OptStepCG2 = 1000
        OptStepConf = 25
        NumberConf = 1
        WeightedSearch = False

    else:
        OptStepCG1 = 250
        OptStepCG2 = 250
        OptStepConf = 25
        NumberConf = 20  # 100
        WeightedSearch = True

    builder = ob.OBBuilder()
    builder.Build(mol)
    ff = ob.OBForceField.FindForceField("uff")
    ff.Setup(mol)
    ff.ConjugateGradients(OptStepCG1)

    for i in reversed(range(NumConf)):
        # 1. The number of random conformers to consider during the search.
        # 2. The number of steps to take during geometry optimization for each conformer.
        if WeightedSearch is True:
            ff.WeightedRotorSearch(NumberConf, OptStepConf)
        else:
            ff.RandomRotorSearch(NumberConf, OptStepConf)
        ff.ConjugateGradients(OptStepCG2)
        ff.UpdateCoordinates(mol)

        obConversion.SetOutFormat("xyz")
        obConversion.WriteFile(
            mol, out_dir + unit_name + '_N' + str(length) + '_C' + str(i + 1) + '.xyz'
        )

        obConversion.SetOutFormat("pdb")
        obConversion.WriteFile(
            mol, out_dir + unit_name + '_N' + str(length) + '_C' + str(i + 1) + '.pdb'
        )

        unit = pd.read_csv(
            out_dir + unit_name + '_N' + str(length) + '_C' + str(i + 1) + '.xyz',
            header=None,
            skiprows=2,
            delim_whitespace=True,
        )
        gen_molecule_vasp(
            unit_name,
            unit,
            0,
            0,
            Inter_Mol_Dis,
            out_dir + unit_name + '_N' + str(length) + '_C' + str(i + 1) + '.vasp',
        )


# Search a good conformer
# INPUT: ID, mol without Hydrogen atom, row indices of dummy and connecting atoms, directory
# OUTPUT: XYZ coordinates of the optimized molecule
def gen_conf_xyz_vasp(
    unit_name,
    m1,
    out_dir,
    ln,
    Nconf,
    Inter_Mol_Dis,
    IrrStruc,
    NCores_opt,
    OPLS,
    GAFF2,
    atom_typing_,
):
    m2 = Chem.AddHs(m1)
    NAttempt = 10000
    if NCores_opt != 1:
        NAttempt = 1000000

    for i in range(10):
        cids = AllChem.EmbedMultipleConfs(
            m2,
            numConfs=Nconf + 10,
            numThreads=NCores_opt,
            randomSeed=i,
            maxAttempts=NAttempt,
        )

    #   if len(cids) > 0:
    #       break
    n = 0
    for cid in cids:
        n += 1
        AllChem.UFFOptimizeMolecule(m2, confId=cid)
        # AllChem.MMFFOptimizeMolecule(m2, confId=cid)

        outfile_name = out_dir + unit_name + '_N' + str(ln) + '_C' + str(n)

        if IrrStruc is False:
            Chem.MolToPDBFile(
                m2, outfile_name + '.pdb', confId=cid
            )  # Generate pdb file
            Chem.MolToXYZFile(
                m2, outfile_name + '.xyz', confId=cid
            )  # Generate pdb file
        else:
            print(
                "\n",
                outfile_name,
                ": Performing a short MD simulation using PySIMM and LAMMPS ...\n",
            )
            disorder_struc(
                unit_name + '_N' + str(ln) + '_C' + str(n), out_dir, NCores_opt
            )
            print("\n", outfile_name, ": MD simulation normally terminated.\n")

        # Generate OPLS parameter file
        if n == 1 and OPLS is True:
            print(unit_name, ": Generating OPLS parameter file ...")
            if os.path.exists(outfile_name + '.pdb'):
                try:
                    Converter.convert(pdb=outfile_name+'.pdb', resname=outfile_name+'_opls',
                                      charge=0, opt=0, outdir='.')
                    print(unit_name, ": OPLS parameter file generated.")
                except BaseException:
                    print('problem running LigParGen for {}.pdb.'.format(outfile_name))
        if GAFF2 is True:
            get_gaff2(outfile_name, out_dir, atom_typing=atom_typing_)

        unit = pd.read_csv(
            outfile_name + '.xyz', header=None, skiprows=2, delim_whitespace=True,
        )
        gen_molecule_vasp(
            unit_name, unit, 0, 0, Inter_Mol_Dis, outfile_name + '.vasp',
        )

        MDlib.gen_sys_data(
            outfile_name + ".lmp",
            unit,
            "",
            unit[1].min() - Inter_Mol_Dis / 2,
            unit[1].max() + Inter_Mol_Dis / 2,
            unit[2].min() - Inter_Mol_Dis / 2,
            unit[2].max() + Inter_Mol_Dis / 2,
            unit[3].min() - Inter_Mol_Dis / 2,
            unit[3].max() + Inter_Mol_Dis / 2,
            False,
            Inter_Mol_Dis=Inter_Mol_Dis,
        )

        if n == Nconf:
            break

    return len(cids)


# This function generates a VASP input (polymer) file
# INPUT: name of VASP directory, name of a monomer, XYZ-coordinates, row numbers for dummy and
# connecting atoms , chemical name of dummy atom, Serial number
# OUTPUT: Generates a VASP input file
def gen_molecule_vasp(unit_name, unit, atom1, atom2, Inter_Mol_Dis, outVASP):

    if atom1 != 0:
        unit = trans_origin(unit, atom1)
        if atom2 != 0:
            unit = alignZ(unit, atom1, atom2)

    unit = unit.sort_values(by=[0])

    # keep_space = 12
    file = open(outVASP, 'w+')
    file.write('### ' + str(unit_name) + ' ###\n')
    file.write('1\n')
    a_vec = unit[1].max() - unit[1].min() + Inter_Mol_Dis
    b_vec = unit[2].max() - unit[2].min() + Inter_Mol_Dis
    c_vec = unit[3].max() - unit[3].min() + Inter_Mol_Dis

    # move unit to the center of a box
    unit[1] = unit[1] - unit[1].min() + Inter_Mol_Dis / 2
    unit[2] = unit[2] - unit[2].min() + Inter_Mol_Dis / 2
    unit[3] = unit[3] - unit[3].min() + Inter_Mol_Dis / 2

    file.write(' ' + str(a_vec) + ' ' + str(0.0) + ' ' + str(0.0) + '\n')
    file.write(' ' + str(0.0) + ' ' + str(b_vec) + ' ' + str(0.0) + '\n')
    file.write(' ' + str(0.0) + ' ' + str(0.0) + ' ' + str(c_vec) + '\n')

    ele_list = []
    count_ele_list = []
    for element in sorted(set(unit[0].values)):
        ele_list.append(element)
        count_ele_list.append(list(unit[0].values).count(element))

    for item in ele_list:
        file.write(str(item) + '  ')

    file.write('\n ')
    for item in count_ele_list:
        file.write(str(item) + ' ')

    file.write('\nCartesian\n')

    file.write(unit[[1, 2, 3]].to_string(header=False, index=False))
    file.close()


def disorder_struc(filename, dir_path, NCores_opt):
    from pysimm import system, lmps, forcefield

    # pdb to cml
    obConversion.SetInAndOutFormats("pdb", "cml")
    obConversion.ReadFile(mol, os.path.join(dir_path, filename + '.pdb'))
    obConversion.WriteFile(mol, os.path.join(dir_path, filename + '.cml'))

    # MD simulation followed by opt
    scml = system.read_cml(os.path.join(dir_path, filename + '.cml'))
    scml.apply_forcefield(forcefield.Gaff2())
    lmps.quick_md(scml, np=NCores_opt, ensemble='nvt', timestep=0.5, run=15000)
    lmps.quick_min(scml, np=NCores_opt, etol=1.0e-5, ftol=1.0e-5)

    # Write files
    scml.write_xyz(os.path.join(dir_path, filename + '.xyz'))
    scml.write_pdb(os.path.join(dir_path, filename + '.pdb'))


def opt_mol_ob(
    path_in,
    format_in="xyz",
    format_out="xyz",
    OptStep=10000,
    forcefield="uff",
    OutFile=False,
    path_out='',
):
    obConversion.SetInFormat(format_in)
    mol = ob.OBMol()
    obConversion.ReadFile(mol, path_in)

    ff = ob.OBForceField.FindForceField(forcefield)
    ff.Setup(mol)
    ff.ConjugateGradients(OptStep)
    ff.UpdateCoordinates(mol)

    if OutFile is True:
        obConversion.SetOutFormat(format_out)
        obConversion.WriteFile(mol, path_out)
    return mol, ff.Energy()


def screen_Candidates(
    dir_path_in,
    format_in="POSCAR",
    format_out="POSCAR",
    OptStep=10000,
    forcefield="uff",
    NumCandidate=50,
    NCores_opt=1,
):
    # Ouput directory
    list_dir = dir_path_in.split('/')
    dir_path_out = '/'.join(list_dir[:-1]) + '/' + list_dir[-1] + '_sort'
    build_dir(dir_path_out)

    # List of POSCAR files in the input directory
    vasp_input_list = glob.glob(dir_path_in + "/" + "*.vasp")

    if len(vasp_input_list) == 0:
        return None

    # Optimize all crystal models
    list_cryst = []
    for path_each in vasp_input_list:
        mol, energy = opt_mol_ob(
            path_each, format_in, format_out, OptStep, forcefield, OutFile=False
        )
        list_cryst.append([mol, energy])

    # result = Parallel(n_jobs=NCores_opt)(delayed(opt_mol_ob)(path_each, format_in,
    # format_out, OptStep, forcefield, OutFile=False) for path_each in vasp_input_list)
    # print(result)
    # list_cryst = []
    # for i in result:
    #    list_cryst.append([i[0], i[1]])

    # Create a pandas DataFrame, sort, and then select
    list_cryst = pd.DataFrame(list_cryst, columns=['OBmol', 'ener'])
    list_cryst = list_cryst.sort_values(by='ener', ascending=True).head(NumCandidate)

    digits = len_digit_number(NumCandidate)
    # Generate poscar files for optimized crystal models
    obConversion.SetOutFormat(format_out)
    count = 1
    for index, row in list_cryst.iterrows():
        obConversion.WriteFile(
            row['OBmol'],
            dir_path_out + '/' + 'cryst_out-' + str(count).zfill(digits) + '.vasp',
        )
        count += 1


def del_tmp_files():
    if os.path.exists("plt.pdb"):
        os.remove("plt.pdb")
    if os.path.exists("olog"):
        os.remove("olog")
    if os.path.exists("optzmat"):
        os.remove("optzmat")
    if os.path.exists("slvzmat"):
        os.remove("slvzmat")
    if os.path.exists("pysimm.sim.in"):
        os.remove("pysimm.sim.in")
    if os.path.exists("sum"):
        os.remove("sum")
    if os.path.exists("out"):
        os.remove("out")
    if os.path.exists("log.lammps"):
        os.remove("log.lammps")
    if os.path.exists("clu.pdb"):
        os.remove("clu.pdb")
    if os.path.exists("LL"):
        os.remove("LL")


def get_gaff2(outfile_name, out_dir, atom_typing='pysimm'):
    print("\nGenerating GAFF2 parameter file ...\n")
    r = MDlib.get_coord_from_pdb(outfile_name + ".pdb")
    from pysimm import system, forcefield

    obConversion.SetInAndOutFormats("pdb", "mol2")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, outfile_name + '.pdb')
    obConversion.WriteFile(mol, outfile_name + '.mol2')

    data_fname = outfile_name + '_gaff2.lmp'

    try:
        print("Pysimm working on {}".format(outfile_name + '.mol2'))
        s = system.read_mol2(outfile_name + '.mol2')
    except BaseException:
        print('problem reading {} for Pysimm.'.format(outfile_name + '.mol2'))

    f = forcefield.Gaff2()
    if atom_typing == 'pysimm':
        try:
            print("Pysimm applying force field for {}.".format(outfile_name + '.mol2'))
            s.apply_forcefield(f, charges='gasteiger')
        except BaseException:
            print(
                'Error applying force field with the mol2 file, switch to using cml file.'
            )

            obConversion.SetInAndOutFormats("pdb", "cml")
            mol = ob.OBMol()
            obConversion.ReadFile(mol, outfile_name + '.pdb')
            obConversion.WriteFile(mol, outfile_name + '.cml')

            s = system.read_cml(outfile_name + '.cml')
            for b in s.bonds:
                if b.a.bonds.count == 3 and b.b.bonds.count == 3:
                    b.order = 4
            s.apply_forcefield(f, charges='gasteiger')
    elif atom_typing == 'antechamber':
        print("Antechamber working on {}".format(outfile_name + '.mol2'))
        MDlib.get_type_from_antechamber(s, outfile_name + '.mol2', 'gaff2', f)
        s.pair_style = 'lj'
        s.apply_forcefield(f, charges='gasteiger', skip_ptypes=True)
    else:
        print('Invalid atom typing option, please select pysimm or antechamber.')
    s.write_lammps(data_fname)
    print("\nGAFF2 parameter file generated.")
