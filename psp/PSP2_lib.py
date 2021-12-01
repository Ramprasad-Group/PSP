from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
import re
import pandas as pd
import os
import psp.PSP_lib as bd

obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')


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
        mol1.DeleteAtom(mol1.GetAtom(smi2_Idx[0])) # One H atom deleted from smi1; so -1 is added
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
    return mol1, mol1

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
        mol_list_smi1 = mol_list[:mol_list.index(Id_smi1)]
    else:
        mol_list_smi1 = []

    if Id_smi2 in mol_list:
        mol_list_smi2 = mol_list[:mol_list.index(Id_smi2)]
    else:
        mol_list_smi2 = []

    # Get index for the first row that has smi1 or smi2
    smi1_start_id_list = [pd_Idx[pd_Idx['smi1'] == Id_smi1].head(1).index.values[0]]
    try:
        smi1_start_id_list += [pd_Idx[pd_Idx['smi2'] == Id_smi1].head(1).index.values[0]]
    except:
        pass
    smi1_start_id = min(smi1_start_id_list)

    smi2_start_id_list = [pd_Idx[pd_Idx['smi2'] == Id_smi2].head(1).index.values[0]]
    try:
        smi2_start_id_list += [pd_Idx[pd_Idx['smi1'] == Id_smi2].head(1).index.values[0]]
    except:
        pass
    smi2_start_id = min(smi2_start_id_list)

    # Get pd starting form first smi1(smi2) to the current index
    pd_Idx_smi1_start = pd_Idx[smi1_start_id:pd_Idx_ind.index.values[0]]
    pd_Idx_smi2_start = pd_Idx[smi2_start_id:pd_Idx_ind.index.values[0]]

    count_delH_smi1=0
    count_delH_smi2=0 # 1 is added for removal of H atoms from corresponding smi1/mol1
    smi1_dum,smi1_link = 0,0
    smi2_dum, smi2_link = 0,0

    # Number of H atoms removed that affect smi1 Idx
    for index1, row1 in pd_Idx_smi1_start.iterrows():
        if row1['smi1'] in mol_list_smi1:
            count_delH_smi1 +=1
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
            count_delH_smi2 +=1
        if row2['smi2'] in mol_list_smi2:
            count_delH_smi2 +=1
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
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'] = [x - count_delH_smi1 for x in pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1']]
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'] = [x - count_delH_smi2 for x in
                                                      pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2']]
    # Adjust of Idx for the same molecular fragments
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][0] -= smi1_dum
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag1'][1] -= smi1_link

    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][0] -= smi2_dum
    pd_Idx.at[pd_Idx_ind.index.values[0], 'frag2'][1] -= smi2_link
    return pd_Idx

def build_pn(unit_name,df_smiles,id,smiles,inter_mol_dis,irr_struc,opls,gaff2,GAFF2_atom_typing,ncore_opt,out_dir):
    result = 'FAILURE'

    # location of input XYZ files
    xyz_in_dir = 'work_dir/'+unit_name
    bd.build_dir(xyz_in_dir)

    # Get SMILES
    smiles_each = df_smiles[df_smiles[id] == unit_name][smiles].values[0]

    L_count = len(list(set(re.findall(r"\[L(.*?)]",smiles_each))))
    dum = 'H'

    smi_list = smiles_each.split(".")
    dum_list = []
    smi1_list = []
    smi2_list = []

    for i in range(1, L_count + 1):
        dum_list.append(str(i + 4) + dum)
        smi_link_list_ind = []
        for j in range(len(smi_list)):
            if smi_list[j].find('[L' + str(i) + ']') != -1:  # Contains given substring
                smi_list[j] = smi_list[j].replace('[L' + str(i) + ']', '[' + str(i + 4) + dum + ']', 1)
                smi_link_list_ind.append(j)
        smi1_list.append(smi_link_list_ind[0])
        smi2_list.append(smi_link_list_ind[1])

    # Prepare a DataFrame for Idx
    pd_Idx = pd.DataFrame(dum_list, columns=['dum'])
    pd_Idx['smi1'] = smi1_list
    pd_Idx['smi2'] = smi2_list

    frag1_Idx, frag2_Idx = [], []
    for index, row in pd_Idx.iterrows():
        a, b = get_linking_Idx(Chem.MolFromSmiles(smi_list[row['smi1']]), dum, row['dum'])
        c, d = get_linking_Idx(Chem.MolFromSmiles(smi_list[row['smi2']]), dum, row['dum'])
        frag1_Idx.append([a, b])
        frag2_Idx.append([c, d])
    pd_Idx['frag1'] = frag1_Idx
    pd_Idx['frag2'] = frag2_Idx

    # Get 3D geometry for each fragment
    obConversion.SetInFormat("xyz")
    OBMol_list = []
    for i in range(len(smi_list)):
        get3DfromRDKitmol(Chem.MolFromSmiles(smi_list[i]), 'work_dir/'+unit_name, str(i))
        path_xyz = os.path.join('work_dir/'+unit_name, str(i) + '.xyz')
        mol = ob.OBMol()
        obConversion.ReadFile(mol, path_xyz)
        OBMol_list.append(mol)

    mol_list = []
    for index, row in pd_Idx.iterrows():
        # Add the first atom to the mol_list
        if index == 0:
            mol_list.append(row['smi1'])

        # Combine mol1/smi1 and mol2/smi2
        OBMol_list[row['smi1']], OBMol_list[row['smi2']] = build_network(pd_Idx, row['dum'],
                                                                             OBMol_list[row['smi1']],
                                                                             OBMol_list[row['smi2']], mol_list)

        if index == pd_Idx.index[-1]:
            # Generate Geometry of the polymer network
            out_xyz = os.path.join(out_dir, unit_name+'.xyz')
            obConversion.WriteFile(OBMol_list[row['smi1']], out_xyz)
            result = 'SUCCESS'
        else:
            # Update mol_list
            if row['smi1'] not in mol_list:
                mol_list = [row['smi1']] + mol_list
            if row['smi2'] not in mol_list:
                mol_list = mol_list + [row['smi2']]
    return unit_name, result




