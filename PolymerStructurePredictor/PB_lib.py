import numpy as np
import pandas as pd
import PolymerStructurePredictor.simulated_annealing as an
import math
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import re

# OpenBabel
import openbabel as ob
obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("xyz","xyz")
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)
constraints = ob.OBFFConstraints()
#obConversion = ob.OBConversion()
#obConversion.SetInAndOutFormats("mol", "xyz")

from scipy.spatial.distance import cdist

# This function try to create a directory
def build_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

# This function minimize molecule using UFF forcefield and Steepest Descent method
# INPUT: ID, path and name of XYZ file, row indices of dummy and connecting atoms, name of working directory
# OUTPUT: XYZ coordinates of the optimized molecule
def localopt(unit_name,file_name,dum1,dum2,atom1,atom2,xyz_tmp_dir):
    obConversion.ReadFile(mol, file_name)
    for atom_id in [dum1+1,dum2+1,atom1+1,atom2+1]:
        constraints.AddAtomConstraint(atom_id)

    # Set the constraints
    ff.Setup(mol, constraints)
    ff.SteepestDescent(5000)
    ff.UpdateCoordinates(mol)
    obConversion.WriteFile(mol, xyz_tmp_dir + unit_name + '_opt.xyz')

    # read XYZ file: skip the first two rows
    unit_opt = pd.read_csv(xyz_tmp_dir + unit_name + '_opt.xyz', header=None, skiprows=2, delim_whitespace=True)
    return unit_opt

# This function create XYZ files from SMILES
# INPUT: ID, SMILES, directory name
# OUTPUT: xyz files in 'work_dir', result = DONE/NOT DONE, mol without Hydrogen atom
def smiles_xyz(unit_name,SMILES,dir_xyz):
    try:
        # Get mol(m1) from smiles
        m1=Chem.MolFromSmiles(SMILES)

        # Add H
        m2 = Chem.AddHs(m1)

        # Get 2D coordinates
        AllChem.Compute2DCoords(m2)

        # Make 3D mol
        AllChem.EmbedMolecule(m2)

        # Change title
        m2.SetProp("_Name", unit_name + '   ' + SMILES)

        # Optimize 3D str
        AllChem.UFFOptimizeMolecule(m2,maxIters=200)
        Chem.rdmolfiles.MolToXYZFile(m2, dir_xyz + unit_name + '.xyz')

        result='DONE'
    except:
        result='NOT_DONE'
    return result, m1

# Search a good conformer
# INPUT: ID, mol without Hydrogen atom, row indices of dummy and connecting atoms, directory
# OUTPUT: XYZ coordinates of the optimized molecule
def find_best_conf(unit_name,m1,dum1,dum2,atom1,atom2,xyz_in_dir):
    m2 = Chem.AddHs(m1)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=100)
    cid_list = []
    for cid in cids:
        AllChem.UFFOptimizeMolecule(m2, confId=cid)
        conf = m2.GetConformer(cid)
        ffu = AllChem.UFFGetMoleculeForceField(m2, confId=cid)
        cid_list.append(
            [cid, abs(Chem.rdMolTransforms.GetDihedralDeg(conf, int(dum1), int(atom1), int(atom2), int(dum2))),
             ffu.CalcEnergy()])
    cid_list = pd.DataFrame(cid_list, columns=['cid', 'Dang', 'Energy'])
    cid_list = cid_list.sort_values(by=['Dang'], ascending=False)
    cid_list = cid_list[cid_list['Dang'] > int(cid_list.head(1)['Dang'].values[0]) - 8.0]
    cid_list = cid_list.sort_values(by=['Energy'], ascending=True)
    Chem.rdmolfiles.MolToXYZFile(m2, xyz_in_dir + str(unit_name) + '.xyz', confId=int(cid_list.head(1)['cid'].values[0]))

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
            if bond.GetBeginAtom().GetSymbol() == '*' or bond.GetEndAtom().GetSymbol() =='*':
                bond_type = bond.GetBondType()
 #               print(bond_type)
                break
    return dummy_index, str(bond_type)

# Build an Oligomer
# INPUT: ID, Length of an oligomer to be built, XYZ-coordinates of a monomer, row numbers for dummy and connecting atoms, unit distance
# OUTPUT: XYZ-coordinates of an oligomer and outcome of connectivity check
def build(unit_name,length,unit,dum1,dum2,atom1,atom2,unit_dis):
    add_dis = add_dis_func(unit, atom1, atom2)

    # XYZ positions of atom2
    unit_disX=unit.loc[atom2].values[1]
    unit_disY=unit.loc[atom2].values[2]
    unit_disZ=unit.loc[atom2].values[3] + unit_dis + add_dis

    build=pd.DataFrame() # START A DATAFRAME
    for len in np.arange(1,length+1,1): # EACH UNIT IN A MOLECULE
        # First unit of the oligomer
        if len == 1:
            build=build.append(unit.drop([dum2]), ignore_index=True)

        # Last unit of the Oligomer
        elif len == length:
            unit[1] = unit[1] + unit_disX
            unit[2] = unit[2] + unit_disY
            unit[3] = unit[3] + unit_disZ

            first_unit=build.copy()
            second_unit=unit.drop([dum1]).copy()
            build = build.append(unit.drop([dum1]), ignore_index=True)
            check_connectivity = CheckConnectivity(unit_name,first_unit, second_unit, build)
            if check_connectivity == 'CORRECT':
                # Calculate distance between atoms in first_unit and second_unit
                dist = cdist(first_unit[[1,2,3]].values,second_unit[[1,2,3]].values)
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

                if (dist<1.6).any(): # in angstrom
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
def TwoMonomers_Dimer(unit_name,unit1,unit2,dum1,dum2,atom1,atom2,dum,unit_dis):
    add_dis = add_dis_func(unit1, atom1, atom2)

    unit1 = trans_origin(unit1, dum1)
    unit1 = alignZ(unit1, dum1, dum2)
    unit2 = trans_origin(unit2, dum1)
    unit2 = alignZ(unit2, dum1, dum2)
    # XYZ positions of atom2
    unit_disX=unit1.loc[atom2].values[1]
    unit_disY=unit1.loc[atom2].values[2]
    unit_disZ=unit1.loc[atom2].values[3]+unit_dis+add_dis

    build=pd.DataFrame() # START A DATAFRAME
    build=build.append(unit1.drop([dum2]), ignore_index=True)

    # Move the second unit
    unit2[1] = unit2[1] + unit_disX
    unit2[2] = unit2[2] + unit_disY
    unit2[3] = unit2[3] + unit_disZ

    first_unit=build.copy()
    second_unit=unit2.drop([dum1]).copy()
    build = build.append(unit2.drop([dum1]), ignore_index=True)
    check_connectivity = CheckConnectivity(unit_name,first_unit, second_unit, build)
    if check_connectivity == 'CORRECT':
        # Calculate distance between atoms in first_unit and second_unit
        dist = cdist(first_unit[[1,2,3]].values,second_unit[[1,2,3]].values)
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

        if (dist<1.6).any(): # in angstrom
            check_connectivity = 'WRONG'

        dum1_2nd, atom1_2nd, dum2_2nd, atom2_2nd = dum1,atom1,dum2+unit1.shape[0]-2,atom2+unit1.shape[0]-2

    else:
        dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd = 0,0,0,0

    return build, check_connectivity, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd



# Find row number of atoms associated with an atom
# INPUT: row number of atoms (atoms) and connecting information obtained from Openbabel
# OUTPUT: atoms directly connected with each atom
def findConnectingAtoms(atoms,neigh_atoms_info):
    ConnectingAtoms=[]
    for i in atoms:
        ConnectingAtoms.extend(neigh_atoms_info.loc[i]['NeiAtom'])
    return ConnectingAtoms

# Find bond order between two atoms
# INPUT: Row number of atom1 and atom2, and connectivity information obtained from Openbabel
# OUTPUT: Bond order between atom1 and atom2
def find_bondorder(atom1,rot_atom1,neigh_atoms_info):
    index_rot_atom1=neigh_atoms_info.loc[atom1]['NeiAtom'].index(rot_atom1)
    return neigh_atoms_info.loc[atom1]['BO'][index_rot_atom1]

# Find single bonds and associated atoms
# INPUT: unit, neigh_atoms_info
# OUTPUT: List of atoms with bond_order = 1
def single_bonds(neigh_atoms_info,unit):
    single_bond = []
    for index, row in neigh_atoms_info.iterrows():
        if len(row['NeiAtom']) < 2:
            neigh_atoms_info = neigh_atoms_info.drop(index)

    for index, row in neigh_atoms_info.iterrows():
        for j in row['NeiAtom']:
            bond_order = find_bondorder(index,j, neigh_atoms_info)
            if bond_order == 1:
                single_bond.append([index,j])
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
# INPUT: Row number of an atom (atom), XYZ-coordinates of a molecule (unit), and connectivity information obtained from Openbabel
# OUTPUT: Row numbers of rotating atoms
def findrotgroups(atom,unit,neigh_atoms_info):
    nei_atom = neigh_atoms_info.loc[atom]['NeiAtom']
    neinei_atom={}
    for i in nei_atom:
        neinei_atom[i]=neigh_atoms_info.loc[i]['NeiAtom'].copy() # collect all neighboring atoms for each atom connected to connecting atom 1
        while len(neinei_atom[i]) < unit.values.shape[0]:
            NumConnect=neinei_atom[i].copy()
            if atom in neinei_atom[i]:
                neinei_atom[i].remove(atom)
            neinei_atom[i].extend(list(set(findConnectingAtoms(neinei_atom[i], neigh_atoms_info))))
            if i in neinei_atom[i]:
                neinei_atom[i].remove(i)
            neinei_atom[i]=list(set(neinei_atom[i]))
            if sorted(neinei_atom[i]) == sorted(NumConnect):
                break
    return neinei_atom #rot_groups

# Find out row numbers of atoms which are allowed to rotate
# INPUT: Row numbers of the two rotating atoms, and groups of atoms connected (directly or indirectly) to the first rotating atom
# OUTPUT: Row numbers of atoms allowed to rotate
def search_rot_atoms(atom1,rot_atom1,rot_groups):
    rot_atoms=[] # collect all atoms to be rotated
    for i in list(rot_groups.keys()):
        if i != rot_atom1:
            rot_atoms.extend(rot_groups[i])
    rot_atoms=list(set([atom1,rot_atom1]+rot_atoms+list(rot_groups.keys())))
    return rot_atoms

# Connection information obtained by OpenBabel
# INPUT: XYZ file
# OUTPUT: Valency and connectivity information
def connec_info(unit_name):
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("xyz")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, unit_name)
    checkvalency=[]
    neigh_atoms_info=[]

    for atom in ob.OBMolAtomIter(mol):
        if atom.GetValence() != atom.GetImplicitValence():
            checkvalency.append(atom.GetIndex())
        neigh_atoms = []
        bond_orders = []
        for allatom in ob.OBAtomAtomIter(atom):
            neigh_atoms.append(allatom.GetIndex())
            bond_orders.append(atom.GetBond(allatom).GetBondOrder())
        neigh_atoms_info.append([neigh_atoms,bond_orders])
    neigh_atoms_info = pd.DataFrame(neigh_atoms_info, columns = ['NeiAtom','BO'])
    return checkvalency, neigh_atoms_info

# This function generates a xyz file
# INPUT: Name of a output file and a DataFrame of element names and respective XYZ-coordinates
# OUTPUT: Write a XYZ file
def gen_xyz(filename, unit):
    with open(filename, 'w') as f:
        f.write(str(unit.values.shape[0]))  # NUMBER OF ATOMS
        f.write("\n\n")  # TWO SPACES
        unit.to_csv(f, sep=' ', index=False, header=False)  # XYZ COORDINATES OF NEW MOLECULE

# This function generates a VASP input (polymer) file
# INPUT: name of VASP directory, name of a monomer, XYZ-coordinates, row numbers for dummy and connecting atoms , chemical name of dummy atom, Serial number
# OUTPUT: Generates a VASP input file
def gen_vasp(vasp_dir,unit_name,unit,dum1,dum2,atom1,atom2,dum,unit_dis,SN):
    add_dis = add_dis_func(unit,atom1,atom2)

    unit = trans_origin(unit, atom2)
    unit = alignZ(unit, atom2, dum1)
    unit = unit.sort_values(by=[0])

    keep_space = 6
    file = open(vasp_dir + unit_name.replace('.xyz', '') + '_' + SN + '.vasp', 'w+')
    file.write('### ' + str(unit_name) + ' ###\n')
    file.write('1\n')
    a_vec = unit[1].max() - unit[1].min() + keep_space
    b_vec = unit[2].max() - unit[2].min() + keep_space

    c_vec = unit.loc[dum1][3] + unit_dis + add_dis#

    # move unit to the center of a box
    unit[3] = unit[3] + (1.68 + unit_dis + add_dis)/2

    unit = unit.drop([dum1,dum2])
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

    for index, row in unit.iterrows():
        file.write(' ' + str(row[1] + (a_vec / 2)) + ' ' + str(row[2] + (b_vec / 2)) + ' ' + str(row[3]) + '\n')

    file.close()


# Distance between two points
def distance(x1,x2,x3,y1,y2,y3):
    return np.sqrt((x1-y1)**2+(x2-y2)**2+(x3-y3)**2)

# Angle between two vectors (Vector1: a1,a2; Vector2: b1,b2)
# INPUT: 4 data points with XYZ coordinates
# OUTPUT: Angle in Degree
def angle_vec(a1,a2,b1,b2):
    x1,x2,x3=a2[1]-a1[1],a2[2]-a1[2],a2[3]-a1[3]
    y1,y2,y3=b2[1]-b1[1],b2[2]-b1[2],b2[3]-b1[3]
    ab=x1*y1 + x2*y2 + x3*y3
    mod_a=np.sqrt(x1*x1 + x2*x2 + x3*x3)
    mod_b=np.sqrt(y1*y1 + y2*y2 + y3*y3)
    alpha = math.acos(np.around(ab/(mod_a * mod_b), decimals=15))
    return alpha * (180 / math.pi)

# Translation to origin
# INPUT: XYZ-coordinates and row number of an atom which will be moved to the origin.
# OUTPUT: A new sets of XYZ-coordinates
def trans_origin(unit,atom1): # XYZ coordinates and angle
    unit[1]=unit[1]-(unit.iloc[atom1][1])
    unit[2]=unit[2]-(unit.iloc[atom1][2])
    unit[3]=unit[3]-(unit.iloc[atom1][3])
    return unit

# Align a molecule on Z-axis wrt two atoms
# INPUT: XYZ-coordinates, row numbers of two atoms
# OUTPUT: A new sets of XYZ-coordinates
def alignZ(unit,atom1,atom2):
    dis_zx=np.sqrt((unit.iloc[atom1].values[3]-unit.iloc[atom2].values[3])**2+(unit.iloc[atom1].values[1]-unit.iloc[atom2].values[1])**2)
    angle_zx=(np.arccos(unit.iloc[atom2].values[3]/dis_zx))*180.0/np.pi
    if unit.iloc[atom2].values[1] > 0.0:# or angle_zx < 90.0: # check and improve
        angle_zx = -angle_zx
    unit=rotateXZ(unit, angle_zx)

    dis_zy=np.sqrt((unit.iloc[atom1].values[3]-unit.iloc[atom2].values[3])**2+(unit.iloc[atom1].values[2]-unit.iloc[atom2].values[2])**2)
    angle_zy=(np.arccos(unit.iloc[atom2].values[3]/dis_zy))*180.0/np.pi
    if unit.iloc[atom2].values[2] > 0.0:# or angle_zy < 90.0: # need to improve
        angle_zy = -angle_zy

    unit=rotateYZ(unit, angle_zy)
    return unit

# Rotate Molecule along Z-axis
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateZ(unit,theta,rot_atoms): # XYZ coordinates, angle and atoms to be rotated
    R_z=np.array([[np.cos(theta*np.pi/180.0),-np.sin(theta*np.pi/180.0),0],[np.sin(theta*np.pi/180.0),np.cos(theta*np.pi/180.0),0],[0,0,1]])
    rot_XYZ=unit.loc[(rot_atoms),[1,2,3]].copy()

    XYZcollect=[]
    for eachatom in np.arange(rot_XYZ.values.shape[0]):
        rotate_each=rot_XYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ=pd.DataFrame(XYZcollect, index=rot_atoms, columns=[1,2,3])
    unit.loc[(rot_atoms),[1,2,3]]=newXYZ[[1,2,3]]
    return unit

# Rotate on XZ plane
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateXZ(unit, theta):  # XYZ coordinates and angle
    R_z=np.array([[np.cos(theta*np.pi/180.0),-np.sin(theta*np.pi/180.0)],[np.sin(theta*np.pi/180.0),np.cos(theta*np.pi/180.0)]])
    oldXYZ=unit[[1,3]].copy()
    XYZcollect=[]
    for eachatom in np.arange(oldXYZ.values.shape[0]):
        rotate_each=oldXYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ=pd.DataFrame(XYZcollect)
    unit[[1,3]]=newXYZ[[0,1]]
    return unit

# Rotate on ZY plane
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateYZ(unit, theta):  # XYZ coordinates and angle
    R_z=np.array([[np.cos(theta*np.pi/180.0),-np.sin(theta*np.pi/180.0)],[np.sin(theta*np.pi/180.0),np.cos(theta*np.pi/180.0)]])
    oldXYZ=unit[[2,3]].copy()
    XYZcollect=[]
    for eachatom in np.arange(oldXYZ.values.shape[0]):
        rotate_each=oldXYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ=pd.DataFrame(XYZcollect)
    unit[[2,3]]=newXYZ[[0,1]]
    return unit

# Check connectivity between atoms in a dimer
# INPUT: ID, XYZ-coordinates of unit1, unit2 and dimer.
# OUTPUT: Check connectivity (CORRECT or WRONG)
def CheckConnectivity(unit_name,unit1,unit2,dimer):
    gen_xyz('work_dir/unit1_' + unit_name + '.xyz', unit1)
    gen_xyz('work_dir/unit2_' + unit_name +'.xyz', unit2)
    gen_xyz('work_dir/dimer_' + unit_name + '.xyz', dimer)

    check_valency_unit1, neigh_atoms_info_unit1 = connec_info('work_dir/unit1_' + unit_name + '.xyz')
    check_valency_unit2, neigh_atoms_info_unit2 = connec_info('work_dir/unit2_' + unit_name + '.xyz')
    check_valency_dimer, neigh_atoms_info_dimer = connec_info('work_dir/dimer_' + unit_name + '.xyz')

    Num_atoms_unit1=len(neigh_atoms_info_unit1.index.tolist())
    for index, row in neigh_atoms_info_unit2.iterrows():
        row['NeiAtom'] = [x + Num_atoms_unit1 for x in row['NeiAtom']]
    neigh_atoms_info_ideal_dimer = pd.concat([neigh_atoms_info_unit1,neigh_atoms_info_unit2])

    count=0
    check_connectivity = 'CORRECT'
    for row in range(len(neigh_atoms_info_ideal_dimer.index.tolist())):
        if sorted(neigh_atoms_info_ideal_dimer.iloc[row]['NeiAtom']) != sorted(neigh_atoms_info_dimer.iloc[row]['NeiAtom']):
            count += 1
            # If a atom in first unit is connected to more than one atom of the second unit; reject the structure
            if len(neigh_atoms_info_dimer.iloc[row]['NeiAtom'])-len(neigh_atoms_info_ideal_dimer.iloc[row]['NeiAtom']) > 1:
                check_connectivity = 'WRONG'
                break
            # if connectivity information is changed for more than two atoms, reject the structure
            elif count > 2:
                check_connectivity = 'WRONG'
                break
    return check_connectivity

# This function rotate a molecule; translate to origin, align on the Z-axis, rotate around Z-axis
# INPUT: XYZ coordinate, row numbers of two atoms, Atoms involve in rotation, angle
# OUTPUT: Rotated geometry
def rot_unit(unit,atom1,rot_atom1,rot_atoms,angle):
    unit = trans_origin(unit, atom1)
    unit = alignZ(unit, atom1, rot_atom1)
    unit = rotateZ(unit, angle, rot_atoms)
    return unit

# This function estimate distance between two repeating units
# INPUT: XYZ coordinates, row numbers for connecting atoms
# OUTPUT: Distance
def add_dis_func(unit,atom1,atom2):
    add_dis = 0.0
    if unit.loc[atom1][0] == 'C' and unit.loc[atom2][0] == 'N':
        add_dis = -0.207
    elif unit.loc[atom1][0] == 'N' and unit.loc[atom2][0] == 'N':
        add_dis = -0.3
    elif unit.loc[atom1][0] == 'C' and unit.loc[atom2][0] == 'O':
        add_dis = -0.223
    elif unit.loc[atom1][0] == 'O' and unit.loc[atom2][0] == 'O':
        add_dis = -0.223
    return add_dis

# Rotate over Z-axis and combine it with another unit to build new monomer unit (non periodic)
def build_dimer_rotate(unit_name,rot_angles,unit1,unit2,dum,dum1,dum2,atom1,atom2,unit_dis):
    add_dis = add_dis_func(unit1,atom1,atom2)
    list_conf=[]
    unit_2nd = {}  # All possible new units
    count = 1
    unit_dimer=pd.DataFrame()

    unit1 = trans_origin(unit1, atom1)
    unit1 = alignZ(unit1, atom1, dum1)

    for i in rot_angles:
        unit_2nd[count] = unit2.copy()
        unit_2nd[count] = trans_origin(unit_2nd[count], atom1)
        unit_2nd[count] = alignZ(unit_2nd[count], atom1,atom2)
        unit_2nd[count] = rotateZ(unit_2nd[count],i,np.arange(len(unit_2nd[count][0].values)))
        # combine two units
        unit_2nd[count] = trans_origin(unit_2nd[count], dum2)
        unit_2nd[count] = alignZ(unit_2nd[count], dum2, atom2)
        unit_2nd[count] = trans_origin(unit_2nd[count], atom2)
        unit_2nd[count][3] = unit_2nd[count][3] + unit1[3][dum1] + unit_dis + add_dis
        unit_2nd[count] = unit_2nd[count].drop([dum2])
        unit_2nd[count] = unit_2nd[count].append(unit1.drop([dum1]), ignore_index=True)

        gen_xyz('work_dir/' + unit_name + '.xyz', unit_2nd[count])
        check_valency, neigh_atoms_info_dimer = connec_info('work_dir/' + unit_name + '.xyz')

        dum1_2nd, atom1_2nd, dum2_2nd, atom2_2nd = dum2+unit1.shape[0]-2, atom2+unit1.shape[0]-2, dum1, atom1

        Num_atoms_unit = len(unit_2nd[count].index.tolist())
        check_connectivity = CheckConnectivity(unit_name, unit_2nd[count].iloc[0:int(Num_atoms_unit/2)],unit_2nd[count].iloc[int(Num_atoms_unit/2):Num_atoms_unit], unit_2nd[count])
        if check_connectivity == 'CORRECT':
            # Distance between two dummy atoms and angle between two vectors originated from dummy and connecting atoms
            dis_dum1_dum2 = distance(unit_2nd[count][1][dum1_2nd], unit_2nd[count][2][dum1_2nd], unit_2nd[count][3][dum1_2nd], unit_2nd[count][1][dum2_2nd],
                                     unit_2nd[count][2][dum2_2nd], unit_2nd[count][3][dum2_2nd])
            ang_1st_2nd = angle_vec(unit_2nd[count].loc[atom1_2nd], unit_2nd[count].loc[dum1_2nd], unit_2nd[count].loc[atom2_2nd],
                                    unit_2nd[count].loc[dum2_2nd])

            list_conf.append([count,dis_dum1_dum2,ang_1st_2nd])

        count += 1
    try:
        list_conf = pd.DataFrame(list_conf, columns=['count','dis','Dang'])
        list_conf = list_conf.sort_values(by=['Dang','dis'], ascending=False)

        unit_dimer = unit_2nd[list_conf['count'].head(1).values[0]].copy()
    except:
        pass

    return unit_dimer, neigh_atoms_info_dimer, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd

# This function create a conformer
# INPUT: Name of the molecule, step number in SA, XYZ-coordinates, row numbers of two atoms (single bond). neighboring atom information, angle, temporary directory name, row numbers of dummy and connecting atoms
# OUTPUT: Name of the XYZ file, XYZ-coordinates of rotated unit, dis_dum1_dum2, ang_1st_2nd
def create_conformer(unit_name,sl,unit,bond,neigh_atoms_info,angle,xyz_tmp_dir,dum1,dum2,atom1,atom2):
    rot_groups = findrotgroups(bond[0], unit, neigh_atoms_info)
    rot_atoms = search_rot_atoms(bond[0], bond[1], rot_groups)
    conf_unit = rot_unit(unit, bond[0], bond[1], rot_atoms, angle)

    # Distance between two dummy atoms and angle between two vectors originated from dummy and connecting atoms
    dis_dum1_dum2 = distance(conf_unit[1][dum1], conf_unit[2][dum1], conf_unit[3][dum1], conf_unit[1][dum2], conf_unit[2][dum2], conf_unit[3][dum2])
    ang_1st_2nd = angle_vec(conf_unit.loc[atom1], conf_unit.loc[dum1], conf_unit.loc[atom2], conf_unit.loc[dum2])

    file_name=xyz_tmp_dir + unit_name + '__' + str(sl) + '_' + str(bond[0]) + '_' + str(bond[1]) + '_' + str(angle) + '.xyz'
    gen_xyz(file_name, conf_unit)

    check_connectivity = 'CORRECT'
    penalty=0
    check_valency_new, neigh_atoms_info_new = connec_info(file_name)
    for row in neigh_atoms_info.index.tolist():
        if sorted(neigh_atoms_info.loc[row]['NeiAtom']) != sorted(neigh_atoms_info_new.loc[row]['NeiAtom']):
            check_connectivity = 'WRONG'
            penalty=1
    return file_name, conf_unit, dis_dum1_dum2, ang_1st_2nd, penalty

# Build a dimer and check connectivity
# INPUT: ID, XYZ-coordinates, Connectivity in monomer, row numbers of dummy and connecting atoms, unit distance
# OUTPUT: Connectivity in Dimer
def mono2dimer(unit_name,unit_input,check_connectivity_monomer,dum1, dum2, atom1, atom2, unit_dis):
    if check_connectivity_monomer == 'CORRECT':
        unit_copy = unit_input.copy()
        # Build a dimer
        unit_copy = trans_origin(unit_copy, dum1)
        unit_copy = alignZ(unit_copy, dum1, dum2)

        polymer, check_connectivity_dimer = build(unit_name, 2, unit_copy, dum1, dum2, atom1, atom2, unit_dis)
    else:
        check_connectivity_dimer = 'WRONG'

    return check_connectivity_dimer

# Build oligomers
# INPUT: XYZ-coordinates, ID, row numbers of dummy and connecting atoms, length of oligomer, unit distance and information by OpenBabel
def oligomer_build(unit,unit_name,dum1, dum2, atom1, atom2,oligo_len,unit_dis,neigh_atoms_info):
    unit_copy = unit.copy()
    unit_copy = trans_origin(unit_copy, dum1)
    unit_copy = alignZ(unit_copy, dum1, dum2)

    if oligo_len == 1:
        oligomer = unit_copy
        dum2_oligo = dum2
        atom2_oligo = atom2
    else:
        oligomer, check_connectivity_dimer = build(unit_name, oligo_len, unit_copy, dum1, dum2, atom1, atom2, unit_dis)

        add_atoms = (oligo_len - 1) * (len(unit.index) - 2)
        dum2_oligo = dum2 + add_atoms
        atom2_oligo = atom2 + add_atoms

    if find_bondorder(dum1, atom1, neigh_atoms_info) == 1:
        dums=[dum1,dum2_oligo]
        atoms=[atom1,atom2_oligo]
        for i, j in zip(dums, atoms):
            vec1,vec2,vec3 = oligomer.loc[i][1]-oligomer.loc[j][1],oligomer.loc[i][2]-oligomer.loc[j][2],oligomer.loc[i][3]-oligomer.loc[j][3]
            vec_mag = np.sqrt(vec1*vec1+vec2*vec2+vec3*vec3)
            vec_normal = (vec1,vec2,vec3)/vec_mag
            new_coord = (oligomer.loc[j][1],oligomer.loc[j][2],oligomer.loc[j][3]) + 1.08 * vec_normal
            oligomer = oligomer.append(pd.DataFrame(['H']+list(new_coord)).T,ignore_index=True)

    return oligomer, dum1, atom1, dum2_oligo, atom2_oligo


# Build a polymer from a monomer unit. (main function)
# 1. Rotate all single bonds and find out the best monomer by maximizing angle between two possible vectors between two sets of connecting and dummy atoms.
# 2. Distance between two dummy atoms is considered as the second criteria to choose a good monomer unit.
# 3. If a good monomer is found, then build a dimer.
# 4. If a good monomer is not found, combine two monomers (flip/rotate one unit) and consider it as a new monomer unit
# 5. Search the best unit and build a dimer
# 6. Always check connectivity between atoms to verify if a monomer/dimer is not acceptable or not.
# 7. Minimize final geometry using Steepest Descent

def build_polymer(unit_name,df_smiles,ID,xyz_in_dir,xyz_tmp_dir,vasp_out_dir,rot_angles_monomer,rot_angles_dimer,Steps, Substeps,num_conf,length,method):
    vasp_out_dir_indi=vasp_out_dir+unit_name+'/'
    build_dir(vasp_out_dir_indi)

    # Initial values
    decision = 'FAILURE'
    reject_polymers=[]
    SN = 0

    # Get SMILES
    smiles_each = df_smiles[df_smiles[ID] == unit_name]['smiles'].values[0]

    # Get index of dummy atoms and bond type associated with it
    try:
        dum_index, bond_type = FetchDum(smiles_each)
        if len(dum_index) == 2:
            dum1 = dum_index[0]
            dum2 = dum_index[1]
        else:
            print(unit_name, ": There are more than two dummy atoms in the SMILES string; Hint: The PSP works only for one-dimensional polymers.")
            return unit_name, 'REJECT', 0
    except:
        print(unit_name, ": Couldn't fetch the position of dummy atoms. Hints: (1) In SMILES strings, use '*' for a dummy atom, (2) Check RDKit installation.")
        return unit_name, 'REJECT', 0

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
        print(unit_name, ": Unusal bond type (Only single or double bonds are acceptable). Hints: (1) Check bonds between the dummy and connecting atoms in SMILES string (2) Check RDKit installation.")
        return unit_name, 'REJECT', 0

    # Replace '*' with dummy atom
    smiles_each = smiles_each.replace(r'*', dum)

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz, m1 = smiles_xyz(unit_name,smiles_each,xyz_in_dir)

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(unit_name, ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string, (2) Check RDKit installation.")
        return unit_name, 'REJECT', 0

    # read XYZ file: skip the first two rows
    unit=pd.read_csv(xyz_in_dir + unit_name + '.xyz', header=None, skiprows=2, delim_whitespace=True)

    # Collect valency and connecting information for each atom
    check_valency, neigh_atoms_info = connec_info(xyz_in_dir + unit_name + '.xyz')

    try:
    # Find connecting atoms associated with dummy atoms.
    # dum1 and dum2 are connected to atom1 and atom2, respectively.
        atom1=neigh_atoms_info['NeiAtom'][dum1].copy()[0]
        atom2=neigh_atoms_info['NeiAtom'][dum2].copy()[0]

    except:
        print(unit_name, ": Couldn't get the position of connecting atoms. Hints: (1) XYZ coordinates are not acceptable, (2) Check Open Babel installation.")
        return unit_name, 'REJECT', 0

    if atom1 == atom2:
        print(unit_name, ": Both dummy atoms connect to the same connecting atom. Hint: (1) the PSP can't handle this")
        return unit_name, 'REJECT', 0

    else:
        # create 100 conformers and select which has the largest dihedral angle (within 8 degree) and lowest energy
        find_best_conf(unit_name,m1,dum1,dum2,atom1,atom2,xyz_in_dir)

        # Minimize geometry using steepest descent
        unit = localopt(unit_name,xyz_in_dir + unit_name + '.xyz', dum1, dum2, atom1, atom2, xyz_tmp_dir)

        check_connectivity_dimer = mono2dimer(unit_name, unit, 'CORRECT', dum1, dum2, atom1, atom2, unit_dis)

        if check_connectivity_dimer == 'CORRECT':
            decision = 'SUCCESS'
            SN += 1

            if 'n' in length:
                gen_vasp(vasp_out_dir_indi, unit_name, unit, dum1, dum2, atom1, atom2, dum, unit_dis, str(SN))

            if len(oligo_list) > 0:
                for oligo_len in oligo_list:
                    oligomer, dum1_oligo, atom1_oligo, dum2_oligo, atom2_oligo = oligomer_build(unit, unit_name, dum1, dum2, atom1, atom2, oligo_len, unit_dis,neigh_atoms_info)
                    gen_vasp(vasp_out_dir_indi, unit_name, oligomer, dum1_oligo, dum2_oligo, atom1_oligo, atom2_oligo, dum, unit_dis+5.0, str(SN)+'_N'+str(oligo_len))

                    # Remove dummy atoms
                    oligomer = oligomer.drop([dum1_oligo,dum2_oligo])
                    gen_xyz(vasp_out_dir_indi+unit_name+'_'+str(SN)+'_N'+str(oligo_len)+'.xyz',oligomer)

            if SN == num_conf:
                return unit_name, 'SUCCESS', 1

        # Simulated Annealing
        if method == 'SA' and SN < num_conf:
            ##  Find single bonds and rotate
            single_bond = single_bonds(neigh_atoms_info,unit)

            results = an.SA(unit_name,unit,single_bond,rot_angles_monomer,neigh_atoms_info,xyz_tmp_dir,dum1,dum2,atom1,atom2,Steps,Substeps)
            results = results.sort_index(ascending=False)

            # Keep num_conf+10 rows to reduce computational costs
            results = results.head(num_conf+10)
            TotalSN=0
            for index, row in results.iterrows():
                TotalSN +=1

                # Minimize geometry using steepest descent
                final_unit = localopt(unit_name,row['xyzFile'], dum1, dum2, atom1, atom2, xyz_tmp_dir)

                check_connectivity_monomer = 'CORRECT'

                check_valency_new, neigh_atoms_info_new = connec_info(row['xyzFile'])
                for row in neigh_atoms_info.index.tolist():
                    if sorted(neigh_atoms_info.loc[row]['NeiAtom']) != sorted(neigh_atoms_info_new.loc[row]['NeiAtom']):
                        check_connectivity_monomer = 'WRONG'

                check_connectivity_dimer = mono2dimer(unit_name, final_unit, check_connectivity_monomer, dum1, dum2, atom1, atom2,unit_dis)

                if check_connectivity_dimer == 'CORRECT':
                    decision='SUCCESS'
                    SN += 1
                    if 'n' in length:
                        gen_vasp(vasp_out_dir_indi,unit_name,final_unit,dum1,dum2,atom1,atom2,dum,unit_dis,str(SN))

                    if len(oligo_list) > 0:
                        for oligo_len in oligo_list:
                            oligomer, dum1_oligo, atom1_oligo, dum2_oligo, atom2_oligo = oligomer_build(unit, unit_name,dum1, dum2,atom1, atom2,oligo_len,unit_dis,neigh_atoms_info)
                            gen_vasp(vasp_out_dir_indi, unit_name, oligomer, dum1_oligo, dum2_oligo, atom1_oligo,atom2_oligo, dum, unit_dis + 5.0, str(SN) + '_N' + str(oligo_len))

                            # Remove dummy atoms
                            oligomer = oligomer.drop([dum1_oligo,dum2_oligo])
                            gen_xyz(vasp_out_dir_indi + unit_name + '_' + str(SN) + '_N' + str(oligo_len) + '.xyz', oligomer)

                    if SN == num_conf:
                         break

                # If we do not get a proper monomer unit, then consider a dimer as a monomer unit and build a dimer of the same
                if SN < num_conf and TotalSN == results.index.size:
                    unit = final_unit.copy()
                    unit = trans_origin(unit, atom1)
                    unit = alignZ(unit, atom1, atom2)

                    unit_dimer, neigh_atoms_info_dimer, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd = build_dimer_rotate(unit_name, rot_angles_dimer, unit, unit, dum,dum1, dum2, atom1, atom2, unit_dis)
                    isempty = unit_dimer.empty

                    if isempty == True:
                        # print(unit_name, ": Couldn't find an acceptable dimer. Please check the structure.")
                        return unit_name, 'FAILURE', 0

                    # Generate XYZ file
                    gen_xyz(xyz_tmp_dir + unit_name +'_dimer.xyz', unit_dimer)

                    # Minimize geometry using steepest descent
                    unit_dimer = localopt(unit_name,xyz_tmp_dir + unit_name +'_dimer.xyz', dum1, dum2, atom1, atom2, xyz_tmp_dir)

                    check_connectivity_dimer = mono2dimer(unit_name, unit_dimer, 'CORRECT', dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd,unit_dis)

                    if check_connectivity_dimer == 'CORRECT':
                        decision = 'SUCCESS'
                        SN += 1
                        if 'n' in length:
                            gen_vasp(vasp_out_dir_indi, unit_name, unit_dimer, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd, dum, unit_dis,str(SN))

                        if len(oligo_list) > 0:
                            for oligo_len in oligo_list:
                                oligomer, dum1_oligo, atom1_oligo, dum2_oligo, atom2_oligo = oligomer_build(unit,unit_name,dum1, dum2,atom1,atom2,oligo_len,unit_dis,neigh_atoms_info)
                                gen_vasp(vasp_out_dir_indi, unit_name, oligomer, dum1_oligo, dum2_oligo, atom1_oligo,atom2_oligo, dum, unit_dis + 5.0, str(SN) + '_N' + str(oligo_len))

                                # Remove dummy atoms
                                oligomer = oligomer.drop([dum1_oligo,dum2_oligo])
                                gen_xyz(vasp_out_dir_indi + unit_name + '_' + str(SN) + '_N' + str(oligo_len) + '.xyz',oligomer)

                        if SN == num_conf:
                            break
                    ##  Find single bonds and rotate
                    single_bond_dimer = single_bonds(neigh_atoms_info_dimer,unit_dimer)

                    results = an.SA(unit_name,unit_dimer, single_bond_dimer, rot_angles_monomer, neigh_atoms_info_dimer, xyz_tmp_dir, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd,Steps, Substeps)
                    results = results.sort_index(ascending=False)

                    for index, row in results.iterrows():
                        # Minimize geometry using steepest descent
                        final_unit = localopt(unit_name,row['xyzFile'], dum1, dum2, atom1, atom2, xyz_tmp_dir)

                        check_connectivity_monomer = 'CORRECT'

                        check_valency_new, neigh_atoms_info_new = connec_info(row['xyzFile'])
                        for row in neigh_atoms_info.index.tolist():
                            if sorted(neigh_atoms_info_dimer.loc[row]['NeiAtom']) != sorted(neigh_atoms_info_new.loc[row]['NeiAtom']):
                                check_connectivity_monomer = 'WRONG'

                        check_connectivity_dimer = mono2dimer(unit_name, final_unit, check_connectivity_monomer, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd,unit_dis)

                        if check_connectivity_dimer == 'CORRECT':
                            decision='SUCCESS'
                            SN += 1
                            if 'n' in length:
                                gen_vasp(vasp_out_dir_indi, unit_name, final_unit, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd, dum,unit_dis,str(SN))

                            if len(oligo_list) > 0:
                                for oligo_len in oligo_list:
                                    oligomer, dum1_oligo, atom1_oligo, dum2_oligo, atom2_oligo = oligomer_build(unit,unit_name,dum1,dum2,atom1,atom2,oligo_len,unit_dis,neigh_atoms_info)
                                    gen_vasp(vasp_out_dir_indi, unit_name, oligomer, dum1_oligo, dum2_oligo,atom1_oligo, atom2_oligo, dum, unit_dis + 5.0,str(SN) + '_N' + str(oligo_len))

                                    # Remove dummy atoms
                                    oligomer = oligomer.drop([dum1_oligo,dum2_oligo])
                                    gen_xyz(
                                        vasp_out_dir_indi + unit_name + '_' + str(SN) + '_N' + str(oligo_len) + '.xyz',oligomer)

                            if SN == num_conf:
                                break

        elif method == 'Dimer' and SN < num_conf:
            SN=0
            for angle in rot_angles_dimer:
                unit1 = unit.copy()
                unit2 = unit.copy()
                unit2 = trans_origin(unit2, atom1)
                unit2 = alignZ(unit2, atom1, atom2)
                unit2 = rotateZ(unit2, angle, np.arange(len(unit2[0].values)))

                dimer, check_connectivity_monomer, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd = TwoMonomers_Dimer(unit_name, unit1, unit2, dum1, dum2, atom1, atom2, dum, unit_dis)
                if atom1_2nd != atom2_2nd:
                    if check_connectivity_monomer == 'CORRECT':
                        unit = dimer.copy()
                        # Build a dimer
                        unit = trans_origin(unit, dum1_2nd)
                        unit = alignZ(unit, dum1_2nd, dum2_2nd)
                        polymer, check_connectivity_dimer = build(unit_name, 2, unit, dum1_2nd, dum2_2nd, atom1_2nd,
                                                                  atom2_2nd,unit_dis)

                        if check_connectivity_dimer == 'CORRECT':
                            decision = 'SUCCESS'
                            SN += 1
                            if 'n' in length:
                                gen_vasp(vasp_out_dir_indi, unit_name, dimer, dum1_2nd, dum2_2nd, atom1_2nd, atom2_2nd, dum,unit_dis,str(SN))

                            if len(oligo_list) > 0:
                                for oligo_len in oligo_list:
                                    oligomer, dum1_oligo, atom1_oligo, dum2_oligo, atom2_oligo = oligomer_build(unit,unit_name,dum1,dum2,atom1,atom2,oligo_len,unit_dis,neigh_atoms_info)
                                    gen_vasp(vasp_out_dir_indi, unit_name, oligomer, dum1_oligo, dum2_oligo,atom1_oligo, atom2_oligo, dum, unit_dis + 5.0,str(SN) + '_N' + str(oligo_len))

                                    # Remove dummy atoms
                                    oligomer = oligomer.drop([dum1_oligo,dum2_oligo])
                                    gen_xyz(
                                        vasp_out_dir_indi + unit_name + '_' + str(SN) + '_N' + str(oligo_len) + '.xyz',oligomer)

                            if SN == num_conf:
                                break

    return unit_name, decision, SN
