from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
import re
import pandas as pd
import os
import psp.PSP_lib as bd
import psp.MD_lib as MDlib
from LigParGenPSP import Converter
import random

obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')


def is_nan(x):
    return x != x


def optimize_geometry(mol1):
    ff.Setup(mol1)
    ff.ConjugateGradients(100000)
    ff.SteepestDescent(100)
    ff.UpdateCoordinates(mol1)
    return mol1


# Get Idx for dummy atom
def get_linking_Idx(m, dum, iso_dum):
    isotope = int(iso_dum.replace(dum, ''))
    for atom in m.GetAtoms():
        if atom.GetSymbol() == dum and atom.GetIsotope() == isotope:
            if len([x.GetIdx() for x in atom.GetNeighbors()]) > 1:
                print("Check your SMILES")
                exit()
            return atom.GetIdx(), [x.GetIdx() for x in atom.GetNeighbors()][0]


def get3DfromRDKitmol(m, dir_xyz, unit_name):
    m2 = Chem.AddHs(m)
    AllChem.Compute2DCoords(m2)
    AllChem.EmbedMolecule(m2)
    AllChem.UFFOptimizeMolecule(m2, maxIters=200)
    Chem.MolToXYZFile(m2, dir_xyz + '/' + unit_name + '.xyz')


def build_network(pd_Idx, dum, mol1, mol2, mol_list):
    builder = ob.OBBuilder()
    builder.SetKeepRings()

    # Dum atom, mol1, mol2, connecting atoms for this particular case
    pd_Idx_ind = pd_Idx[pd_Idx['dum'] == dum]

    # Get mol1 and mol2 Idx
    Id_smi1, Id_smi2 = pd_Idx_ind['smi1'].values[0], pd_Idx_ind['smi2'].values[0]
    pd_Idx = update_df_IDx(pd_Idx, mol1, mol_list, pd_Idx_ind, Id_smi1, Id_smi2)
    smi1_Idx = pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1']
    smi2_Idx = pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2']

    # Combine mol1 and mol2
    if all(x in mol_list for x in [Id_smi1, Id_smi2]):
        mol1.AddBond(smi1_Idx[1] + 1, smi2_Idx[1] + 1, 1)
    else:
        mol1 += mol2
        builder.Connect(mol1, smi1_Idx[1] + 1, smi2_Idx[1] + 1)

    # Delete H atom from mol1/smi1
    mol1.DeleteAtom(mol1.GetAtom(smi1_Idx[0] + 1))  # OpenBabel counts from 1

    # Delete H atom from mol2/smi2
    if smi1_Idx[0] < smi2_Idx[0]:
        mol1.DeleteAtom(
            mol1.GetAtom(smi2_Idx[0])
        )  # One H atom deleted from smi1; so -1 is added
    else:
        mol1.DeleteAtom(mol1.GetAtom(smi2_Idx[0] + 1))

    # Optimize geometry
    ff.Setup(mol1)
    if all(x in mol_list for x in [Id_smi1, Id_smi2]):
        ff.ConjugateGradients(1000)
    else:
        ff.ConjugateGradients(1000)
    ff.SteepestDescent(100)
    ff.UpdateCoordinates(mol1)
    return mol1, mol1, pd_Idx


def update_df_IDx(pd_Idx, mol1, mol_list, pd_Idx_ind, Id_smi1, Id_smi2):
    # Number atoms molecule 1
    n_mol1 = mol1.NumAtoms()

    # Update IDx of mol2
    # Condition 1: if mol2 doesn't exist in the list
    # Add len(mol1) to all mol2 Idx
    if Id_smi2 not in mol_list:
        for index, row in pd_Idx.iterrows():
            if row['smi2'] == Id_smi2:
                pd_Idx.at[index, 'frag2'] = [x + n_mol1 for x in row['frag2']]
            if row['smi1'] == Id_smi2:
                pd_Idx.at[index, 'frag1'] = [x + n_mol1 for x in row['frag1']]

    # Condition 2: if mol1 doesn't exist in the list
    # Add len(mol1) to all Idx already in the list
    # Modification will be performed from the current index # pd_Idx[pd_Idx_ind.index.values[0]:]
    elif Id_smi1 not in mol_list:
        for index, row in pd_Idx.iterrows():
            if row['smi2'] in mol_list:
                pd_Idx.at[index, 'frag2'] = [x + n_mol1 for x in row['frag2']]

    # Find out number of H atoms removed
    if Id_smi1 in mol_list:
        mol_list_smi1 = mol_list[: mol_list.index(Id_smi1)]
    else:
        mol_list_smi1 = []

    if Id_smi2 in mol_list:
        mol_list_smi2 = mol_list[: mol_list.index(Id_smi2)]
    else:
        mol_list_smi2 = []

    # Get index for the first row that has smi1 or smi2
    smi1_start_id_list = [pd_Idx[pd_Idx['smi1'] == Id_smi1].head(1).index.values[0]]
    try:
        smi1_start_id_list += [
            pd_Idx[pd_Idx['smi2'] == Id_smi1].head(1).index.values[0]
        ]
    except Exception:
        pass
    smi1_start_id = min(smi1_start_id_list)

    smi2_start_id_list = [pd_Idx[pd_Idx['smi2'] == Id_smi2].head(1).index.values[0]]
    try:
        smi2_start_id_list += [
            pd_Idx[pd_Idx['smi1'] == Id_smi2].head(1).index.values[0]
        ]
    except Exception:
        pass
    smi2_start_id = min(smi2_start_id_list)

    # Get pd starting form first smi1(smi2) to the current index
    pd_Idx_smi1_start = pd_Idx[smi1_start_id: pd_Idx_ind.index.values[0]]
    pd_Idx_smi2_start = pd_Idx[smi2_start_id: pd_Idx_ind.index.values[0]]

    count_delH_smi1 = 0
    count_delH_smi2 = (
        0  # 1 is added for removal of H atoms from corresponding smi1/mol1
    )
    smi1_dum, smi1_link = 0, 0
    smi2_dum, smi2_link = 0, 0

    # Number of H atoms removed that affect smi1 Idx
    for index1, row1 in pd_Idx_smi1_start.iterrows():
        if row1['smi1'] in mol_list_smi1:
            count_delH_smi1 += 1
        if row1['smi2'] in mol_list_smi1:
            count_delH_smi1 += 1
        if row1['smi1'] == Id_smi1:
            if row1['frag1'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][0]:
                smi1_dum += 1
            if row1['frag1'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][1]:
                smi1_link += 1
        if row1['smi2'] == Id_smi1:
            if row1['frag2'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][0]:
                smi1_dum += 1
            if row1['frag2'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][1]:
                smi1_link += 1

    # Number of H atoms removed that affect smi2 Idx
    for index2, row2 in pd_Idx_smi2_start.iterrows():
        if row2['smi1'] in mol_list_smi2:
            count_delH_smi2 += 1
        if row2['smi2'] in mol_list_smi2:
            count_delH_smi2 += 1
        if row2['smi1'] == Id_smi2:
            if row2['frag1'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][0]:
                smi2_dum += 1
            if row2['frag1'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][1]:
                smi2_link += 1
        if row2['smi2'] == Id_smi2:
            if row2['frag2'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][0]:
                smi2_dum += 1
            if row2['frag2'][0] < pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][1]:
                smi2_link += 1

    # Adjust of Idx for other molecular fragments
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'] = [
        x - count_delH_smi1 for x in pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1']
    ]
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'] = [
        x - count_delH_smi2 for x in pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2']
    ]
    # Adjust of Idx for the same molecular fragments
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][0] -= smi1_dum
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][1] -= smi1_link

    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][0] -= smi2_dum
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][1] -= smi2_link
    return pd_Idx


def build_pn(
    unit_name,
    df_smiles,
    id,
    smiles,
    inter_mol_dis,
    irr_struc,
    opls,
    gaff2,
    GAFF2_atom_typing,
    ncore_opt,
    out_dir,
):
    result = 'FAILURE'
    # location of input XYZ files
    xyz_in_dir = 'work_dir/' + unit_name
    bd.build_dir(xyz_in_dir)

    # Get SMILES
    smiles_each = df_smiles[df_smiles[id] == unit_name][smiles].values[0]

    # Get details of dummy and connecting atoms of all fragments and OBmol of them
    pd_Idx, OBMol_list, OBMol_Mwt = get_Idx_mol(unit_name, smiles_each)

    mol_list = []
    for index, row in pd_Idx.iterrows():
        # Add the first atom to the mol_list
        if index == 0:
            mol_list.append(row['smi1'])

        # Combine mol1/smi1 and mol2/smi2
        OBMol_list[row['smi1']], OBMol_list[row['smi2']], pd_Idx = build_network(
            pd_Idx,
            row['dum'],
            OBMol_list[row['smi1']],
            OBMol_list[row['smi2']],
            mol_list,
        )

        if index == pd_Idx.index[-1]:
            # Generate Geometry of the polymer network
            out_xyz = os.path.join(out_dir, unit_name + '.xyz')
            obConversion.WriteFile(OBMol_list[row['smi1']], out_xyz)
            result = 'SUCCESS'
        else:
            # Update mol_list
            if row['smi1'] not in mol_list:
                mol_list = [row['smi1']] + mol_list
            if row['smi2'] not in mol_list:
                mol_list = mol_list + [row['smi2']]
    return unit_name, result


def get_Idx_mol(unit_name, smiles_each, Mwt_polymer=0):
    L_count = len(list(set(re.findall(r"\[L(.*?)]", smiles_each))))
    dum = 'H'

    smi_list = smiles_each.split(".")
    dum_list = []
    smi1_list = []
    smi2_list = []

    for i in range(1, L_count + 1):
        dum_list.append(str(i + 4) + dum)
        smi_link_list_ind = []
        smi_copy = []
        for j in range(len(smi_list)):
            if smi_list[j].find('[L' + str(i) + ']') != -1:  # Contains given substring
                smi_copy.append(smi_list[j])  # saved to varify the input SMILES
                smi_list[j] = smi_list[j].replace(
                    '[L' + str(i) + ']', '[' + str(i + 4) + dum + ']', 1
                )
                smi_link_list_ind.append(j)
        smi1_list.append(smi_link_list_ind[0])
        if len(smi_link_list_ind) == 2:
            smi2_list.append(smi_link_list_ind[1])
        elif len(smi_copy) == 1 and smi_copy[0].count('[L' + str(i) + ']') == 1:
            smi2_list.append(None)
        else:
            print("Error in SMILES: (1) The same dummy atom found in > 2 places")
            print(
                "               : (2) The same dummy atom found in >= 2 places in a single SMILES"
            )
            print("               : Each dummy atom defines a bond between two atoms.")
            exit()

    # Prepare a DataFrame for Idx
    pd_Idx = pd.DataFrame(dum_list, columns=['dum'])
    pd_Idx['smi1'] = smi1_list
    pd_Idx['smi2'] = smi2_list

    frag1_Idx, frag2_Idx = [], []
    for index, row in pd_Idx.iterrows():
        a, b = get_linking_Idx(
            Chem.MolFromSmiles(smi_list[row['smi1']]), dum, row['dum']
        )
        frag1_Idx.append([a, b])
        if row['smi2'] is not None:
            c, d = get_linking_Idx(
                Chem.MolFromSmiles(smi_list[row['smi2']]), dum, row['dum']
            )
            frag2_Idx.append([c, d])
        else:
            frag2_Idx.append([None, None])
    pd_Idx['frag1'] = frag1_Idx
    pd_Idx['frag2'] = frag2_Idx

    # Get 3D geometry for each fragment
    obConversion.SetInFormat("xyz")
    OBMol_list = []
    OBMol_Mwt = []
    for i in range(len(smi_list)):
        get3DfromRDKitmol(
            Chem.MolFromSmiles(smi_list[i]), 'work_dir/' + unit_name, str(i)
        )
        path_xyz = os.path.join('work_dir/' + unit_name, str(i) + '.xyz')
        mol = ob.OBMol()
        obConversion.ReadFile(mol, path_xyz)
        OBMol_list.append(mol)
        if Mwt_polymer != 0:
            OBMol_Mwt.append(mol.GetMolWt())
    return pd_Idx, OBMol_list, OBMol_Mwt


def end_cap(unit_name, OBMol, link1, link2, leftcap, rightcap):
    obConversion.SetInFormat("xyz")
    builder = ob.OBBuilder()
    builder.SetKeepRings()

    leftcap = leftcap.replace('[*]', '[' + str(5) + 'H' + ']', 1)
    rightcap = rightcap.replace('[*]', '[' + str(5) + 'H' + ']', 1)

    a_left, b_left = get_linking_Idx(Chem.MolFromSmiles(leftcap), 'H', '5H')
    a_right, b_right = get_linking_Idx(Chem.MolFromSmiles(rightcap), 'H', '5H')

    # Generate OBmol
    get3DfromRDKitmol(Chem.MolFromSmiles(leftcap), 'work_dir/' + unit_name, 'leftcap')
    get3DfromRDKitmol(Chem.MolFromSmiles(rightcap), 'work_dir/' + unit_name, 'rightcap')

    path_xyz_left = os.path.join('work_dir/' + unit_name, 'leftcap.xyz')
    path_xyz_right = os.path.join('work_dir/' + unit_name, 'rightcap.xyz')

    mol_left = ob.OBMol()
    obConversion.ReadFile(mol_left, path_xyz_left)

    mol_right = ob.OBMol()
    obConversion.ReadFile(mol_right, path_xyz_right)

    # Delete dummy atoms
    mol_left.DeleteAtom(mol_left.GetAtom(a_left + 1))
    mol_right.DeleteAtom(mol_right.GetAtom(a_right + 1))

    # Adjust linking atom IDx
    if a_left < b_left:
        b_left -= 1
    if a_right < b_right:
        b_right -= 1

    # Number atoms in leftcap and oligomer
    n_left = mol_left.NumAtoms()
    n_oligo = OBMol.NumAtoms()

    # Combine all OBMols
    mol_left += OBMol
    mol_left += mol_right

    if link2 < link1:
        link1, link2 = link2, link1

    # Add bonds
    builder.Connect(mol_left, b_left + 1, n_left + link1 + 1)
    builder.Connect(mol_left, n_left + link2 + 1, n_left + n_oligo + b_right + 1)

    return mol_left


def build_copoly(
    unit_name,
    df_smiles,
    ID,
    SMILES,
    LeftCap,
    RightCap,
    Nunits,
    Mwt,
    Copoly_type,
    define_BB,
    Inter_Mol_Dis,
    output_files,
    Loop,
    IrrStruc,
    GAFF2_atom_typing,
    NCores_opt,
    out_dir,
    seed,
):
    result = 'FAILURE'

    # number building blocks
    if Nunits in df_smiles.columns:
        Nunits = df_smiles[df_smiles[ID] == unit_name][Nunits].values.tolist()[0]
        if type(Nunits) == int or type(Nunits) == float:
            Nunits = [Nunits]
        else:
            Nunits = Nunits.strip().replace("[", "").replace("]", "").split(",")
    else:
        Nunits = [1]

    # Check given Molwt in g/mol for the polymer
    if Mwt in df_smiles.columns:
        Mwt = int(df_smiles[df_smiles[ID] == unit_name][Mwt].values[0])
    else:
        Mwt = 0

    if Mwt == 0:
        Nunits = [int(item) for item in Nunits]
    else:
        Nunits = [float(item) for item in Nunits]
        # Normalize against the sum to ensure that the sum is always 1.0
        Nunits = [float(item) / sum(Nunits) * Mwt for item in Nunits]

    # Get SMILES of individual blocks
    smiles_each = df_smiles[df_smiles[ID] == unit_name][SMILES].values[0]

    smiles_dict = {}
    for smi in smiles_each.strip().split(';'):
        smiles_dict[smi.split(':')[0]] = smi.split(':')[1]

    # Generate an OBMol
    OBMol = ob.OBMol()

    # Get details of dummy and connecting atoms of all fragments and OBmol of them
    OBMol_dict = {}
    pd_Idx_dict = {}
    OBMol_Mwt_dict = {}
    for key in smiles_dict:
        xyz_in_dir_key = 'work_dir/' + unit_name + '_' + key
        bd.build_dir(xyz_in_dir_key)
        pd_Idx_dict[key], OBMol_dict[key], OBMol_Mwt_dict[key] = get_Idx_mol(
            unit_name + '_' + key, smiles_dict[key], Mwt_polymer=Mwt,
        )

    #    if Mwt != 0:
    #        list_blocks = list(OBMol_Mwt_dict.keys())
    #        list_blocks.sort() # Ratio defined in Nunits should follow the alphabetical order
    #        for
    #        print(Mwt)
    #        print(Nunits)
    #        print(OBMol_Mwt_dict)
    #        print(list_blocks)
    #    exit()
    # Is Loop?
    if Loop in df_smiles.columns:
        Loop = eval(str(df_smiles[df_smiles[ID] == unit_name][Loop].values[0]))

    if LeftCap in df_smiles.columns:
        smiles_LCap_ = df_smiles[df_smiles[ID] == unit_name][LeftCap].values[0]
        if is_nan(smiles_LCap_):
            smiles_LCap_ = '[H][*]'
    else:
        smiles_LCap_ = '[H][*]'

    if RightCap in df_smiles.columns:
        smiles_RCap_ = df_smiles[df_smiles[ID] == unit_name][RightCap].values[0]
        if is_nan(smiles_RCap_):
            smiles_RCap_ = '[H][*]'
    else:
        smiles_RCap_ = '[H][*]'

    # Copoly_type
    if Copoly_type in df_smiles.columns:
        Copoly_type = df_smiles[df_smiles[ID] == unit_name][Copoly_type].values[0]
    else:
        Copoly_type = 'UserDefined'

    # Define building blocks
    blocks_dict = {}
    if define_BB in df_smiles.columns:
        define_BB_ = df_smiles[df_smiles[ID] == unit_name][define_BB].values[0]
        if define_BB_ == 'R':
            define_BB_ = list(smiles_dict.keys())
            define_BB_.sort()

            # If Mwt is provided, update Nunits
            if Mwt != 0:
                update_Nunits = []
                for i in range(len(define_BB_)):
                    update_Nunits.append(
                        round(Nunits[i] / OBMol_Mwt_dict[define_BB_[i]][0])
                    )
                Nunits = update_Nunits

            random_seq = []
            for i in range(len(define_BB_)):
                random_seq.extend([define_BB_[i]] * Nunits[i])
                if isinstance(seed, int):
                    random.seed(seed)
                random.shuffle(random_seq)
            blocks_dict[0] = random_seq
            print("\n Polymer building blocks are arranged in the following order: ")
            print("", '-'.join(random_seq))
            print("\n")
            Nunits = [1]
        else:
            define_BB_ = define_BB_.split(':')
            total_units = []
            for block in range(len(define_BB_)):
                blocks_dict[block] = define_BB_[block].split('-')
                total_units.extend(define_BB_[block].split('-'))
            # print(OBMol_Mwt_dict)
            # Calculate Mwt of each block repeating unit
            if Mwt != 0:
                unit_list = list(smiles_dict.keys())
                unit_list.sort()
                update_Nunits = []
                for i in range(len(unit_list)):
                    update_Nunits.append(
                        round(Nunits[i] / OBMol_Mwt_dict[unit_list[i]][0])
                    )

                BB_count = []
                for key in blocks_dict:
                    BB_each_count = []
                    for j in blocks_dict[key]:
                        BB_each_count.append(
                            update_Nunits[(ord(j) - 65)] * (1 / total_units.count(j))
                        )
                    BB_count.append(round(min(BB_each_count)))
                Nunits = BB_count
    else:
        define_BB_ = list(smiles_dict.keys())
        define_BB_.sort()
        blocks_dict[0] = define_BB_

    # location of input XYZ files
    xyz_in_dir = 'work_dir/' + unit_name
    bd.build_dir(xyz_in_dir)

    if Copoly_type == 'UserDefined':
        OBMol, first, second = build_user_defined_poly(
            blocks_dict, pd_Idx_dict, OBMol_dict, Nunits
        )
        if Loop:
            OBMol = build_loop(OBMol, first, second)
        else:
            OBMol = end_cap(unit_name, OBMol, first, second, smiles_LCap_, smiles_RCap_)
        OBMol = optimize_geometry(OBMol)
        result = 'SUCCESS'

    if result == 'SUCCESS':
        GenOutput(
            unit_name, OBMol, output_files, out_dir, Inter_Mol_Dis, GAFF2_atom_typing
        )
    return unit_name, result


def GenOutput(
    unitname, OBMol, list_output, output_dir, Inter_Mol_Dis, GAFF2_atom_typing
):
    # Genrate an XYZ file
    obConversion.SetOutFormat('xyz')
    OBMol.SetTitle("Generated by OpenBabel@PSP -- Output format: " + 'xyz')
    out_filename = os.path.join(output_dir, unitname + '.xyz')
    obConversion.WriteFile(OBMol, out_filename)

    # Get Coordinates of the molecules
    unit = pd.read_csv(out_filename, header=None, skiprows=2, delim_whitespace=True)

    # Generate a POSCAR file
    POSCAR_list = ['poscar', 'POSCAR', 'VASP', 'vasp']
    if any(i in POSCAR_list for i in list_output):
        out_filename = os.path.join(output_dir, unitname + '.vasp')
        bd.gen_molecule_vasp(unitname, unit, 0, 0, Inter_Mol_Dis, out_filename)

    # Generate a LAMMPS datafile
    LAMMPS_list = ['LAMMPS', 'lammps', 'lmp']
    if any(i in LAMMPS_list for i in list_output):
        out_filename = os.path.join(output_dir, unitname + '.lmp')
        MDlib.gen_sys_data(
            out_filename,
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

    # Generate a LAMMPS GAFF2 datafile
    GAFF_list = ['GAFF', 'GAFF2', 'gaff', 'gaff2']
    if any(i in GAFF_list for i in list_output):
        out_filename = os.path.join(output_dir, unitname)
        bd.get_gaff2(out_filename, OBMol, atom_typing=GAFF2_atom_typing)

    # Generate an LAMMPS OPLS datafile
    OPLS_list = ['OPLS', 'OPLS-AA', 'opls', 'opls-aa']
    if any(i in OPLS_list for i in list_output):
        gen_opls_data(unitname, OBMol, output_dir, unitname)
        OPLS_list.append('pdb')
        OPLS_list.append('PDB')

    output_generated = (
        POSCAR_list + LAMMPS_list + GAFF_list + OPLS_list + ['xyz', 'XYZ']
    )
    for i in output_generated:
        if i in list_output:
            list_output.remove(i)
    GenOutputOB(unitname, OBMol, list_output, output_dir)


def gen_opls_data(unit_name, OBMol, output_dir, unitname):
    # Generate a pdb file
    obConversion.SetOutFormat('pdb')
    OBMol.SetTitle("Generated by OpenBabel@PSP -- Output format: " + 'pdb')
    out_filename_pdb = os.path.join(output_dir, unitname + '.pdb')
    obConversion.WriteFile(OBMol, out_filename_pdb)

    # Generate an opls parameter file
    print(unit_name, ": Generating OPLS parameter file ...")
    out_filename = os.path.join(output_dir, unitname + '_opls')
    try:
        Converter.convert(
            pdb=out_filename_pdb, resname=out_filename, charge=0, opt=0, outdir='.'
        )
        print(unit_name, ": OPLS parameter file generated.")
    except BaseException:
        print('problem running LigParGen for :', out_filename_pdb)


def GenOutputOB(unitname, OBMol, list_OB_output, output_dir):
    for i in list_OB_output:
        obConversion.SetOutFormat(i)
        OBMol.SetTitle("Generated by OpenBabel@PSP -- Output format: " + i)
        out_filename = os.path.join(output_dir, unitname + '.' + i)
        obConversion.WriteFile(OBMol, out_filename)


def build_loop(OBMol, first, second):
    OBMol.AddBond(first + 1, second + 1, 1)
    return OBMol


def combine_AB(Amol, Bmol, second_A, first_B, second_B):
    builder = ob.OBBuilder()
    builder.SetKeepRings()

    # Number atoms in leftcap and oligomer
    n_Amol = Amol.NumAtoms()

    Amol += Bmol
    builder.Connect(Amol, second_A + 1, first_B + n_Amol + 1)
    return Amol, second_B + n_Amol


def combine_A_ntimes(OBMol, first, second, n):
    builder = ob.OBBuilder()
    builder.SetKeepRings()
    # Number atoms in leftcap and oligomer
    n_OBmol = OBMol.NumAtoms()

    # Reorder first and second linking atoms
    if second < first:
        first, second = second, first

    main_obmol = ob.OBMol()
    main_obmol += OBMol
    for i in range(n - 1):
        main_obmol += OBMol
        builder.Connect(main_obmol, second + 1, first + (i + 1) * n_OBmol + 1)
        second = second + n_OBmol

    return main_obmol, first, second


def remove_H_get_links(df, OBMol):
    OBMol_copy = ob.OBMol()  # otherwise it will update the orginal OBMol
    OBMol_copy += OBMol

    # Delete dummy H atoms
    if df['frag1'][0][0] > df['frag1'][1][0]:
        OBMol_copy.DeleteAtom(OBMol_copy.GetAtom(df['frag1'][0][0] + 1))
        OBMol_copy.DeleteAtom(OBMol_copy.GetAtom(df['frag1'][1][0] + 1))
    else:
        OBMol_copy.DeleteAtom(OBMol_copy.GetAtom(df['frag1'][1][0] + 1))
        OBMol_copy.DeleteAtom(OBMol_copy.GetAtom(df['frag1'][0][0] + 1))

    # Find out first and second linking atoms
    if df['frag1'][0][1] < df['frag1'][1][1]:
        first_atom = df['frag1'][0][1]
        second_atom = df['frag1'][1][1]
    else:
        first_atom = df['frag1'][1][1]
        second_atom = df['frag1'][0][1]

    # Adjust IDx of the first linking atom
    if first_atom > df['frag1'][0][0] and first_atom > df['frag1'][1][0]:
        first_atom -= 2
    elif first_atom > df['frag1'][0][0] or first_atom > df['frag1'][1][0]:
        first_atom -= 1

    # Adjust IDx of the second linking atom
    if second_atom > df['frag1'][0][0] and second_atom > df['frag1'][1][0]:
        second_atom -= 2
    elif second_atom > df['frag1'][0][0] or second_atom > df['frag1'][1][0]:
        second_atom -= 1

    if second_atom < first_atom:
        first_atom, second_atom = second_atom, first_atom

    return OBMol_copy, first_atom, second_atom


def build_user_defined_poly(blocks_dict, pd_Idx_dict, OBMol_dict, Nunits):
    final_obmol = ob.OBMol()
    count_blocks, blockA_first, blockA_second = 0, 0, 0
    for key in blocks_dict:
        count_blocks += 1
        main_obmol, first_A, second_A = remove_H_get_links(
            pd_Idx_dict[blocks_dict[key][0]], OBMol_dict[blocks_dict[key][0]][0]
        )  # NOTE: it doesn't accept '.' in SMILES
        if len(blocks_dict[key]) > 1:
            for i in range(1, len(blocks_dict[key])):
                obmol, first_B, second_B = remove_H_get_links(
                    pd_Idx_dict[blocks_dict[key][i]], OBMol_dict[blocks_dict[key][i]][0]
                )
                main_obmol, second_A = combine_AB(
                    main_obmol, obmol, second_A, first_B, second_B
                )
        main_obmol, first, second = combine_A_ntimes(
            main_obmol, first_A, second_A, Nunits[key]
        )

        # Connect blocks
        if count_blocks > 1:
            final_obmol, blockA_second = combine_AB(
                final_obmol, main_obmol, blockA_second, first, second
            )
        else:
            final_obmol += main_obmol
            blockA_first = first
            blockA_second = second
    return final_obmol, blockA_first, blockA_second
