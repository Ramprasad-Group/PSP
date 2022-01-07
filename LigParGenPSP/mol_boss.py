# THIS IS THE HEART OF BCC CORRECTION METHODOLOGY
# THIS MODULE DOES THE BCC ASSIGNMENT BY COLLECTING
# BONDING INFO AND ASSIGNING BCC CORRECTIONS FOR ATOMS
import numpy as np


def rev_bnd(bnd):
    a, b = bnd.split('-')
    return b + '-' + a


def sign_bnd(bnd, at):
    if bnd == rev_bnd(bnd):
        si = 0
    else:
        si = (-2 * bnd.split('-').index(at)) + 1
    return si


def get_bcc_types(db, cha, bond):
    rtij = []
    mtij = []
    bond['NTIJ'] = [
        str(cha.TY[i - 1]) + '-' + str(cha.TY[j - 1]) for (i, j) in zip(bond.I, bond.J)
    ]
    for i in bond.NTIJ:
        if i == rev_bnd(i):
            mtij.append('X-X')
            rtij.append(i)
        elif i in db.keys():
            rtij.append(i)
            mtij.append(i)
        elif rev_bnd(i) in db.keys():
            rtij.append(rev_bnd(i))
            mtij.append(rev_bnd(i))
        else:
            print('%5s not found in bonds.csv' % i)
            mtij.append('U-U')
            rtij.append('U-U')
    bond['TIJ'] = mtij
    bond['MTIJ'] = rtij
    bond['AI'] = [str(cha.TY[i - 1]) for i in bond.I]
    bond['AJ'] = [str(cha.TY[j - 1]) for j in bond.J]
    bond['SI'] = [sign_bnd(bnd, at) for bnd, at in zip(bond.TIJ, bond.AI)]
    bond['SJ'] = [sign_bnd(bnd, at) for bnd, at in zip(bond.TIJ, bond.AJ)]
    return bond


def new_mol_info(db, cha, bond):
    #    cha = pd.read_csv('CM1AQ', header=None, delim_whitespace=True)
    #    cha.columns = ['TY', 'Q']
    bond = get_bcc_types(db, cha, bond)
    MOLBtype = {}
    for an in cha.index:
        MOLBtype[an] = list(bond[bond['I'] == an + 1].TIJ) + list(
            bond[bond['J'] == an + 1].TIJ
        )
        if (cha.TY[an] == 'OS') and ('C-OS' in MOLBtype[an]):
            print("Changing OS TO OE")
            cha.loc[an, 'TY'] = 'OE'
            bond = get_bcc_types(db, cha, bond)
        # Seperate Correction for Esters
        if (cha.TY[an] == 'C') and ('C-O' in MOLBtype[an]):
            if ('C-OS' in MOLBtype[an]) or ('C-OE' in MOLBtype[an]):
                print("Changing OS TO OE")
                cha.loc[an, 'TY'] = 'CE'
                bond = get_bcc_types(db, cha, bond)
        # Seperate Correction for Amides
        if (cha.TY[an] == 'C') and ('C-N' in MOLBtype[an]):
            print("Changing C TO CAM")
            cha.loc[an, 'TY'] = 'CAM'
            bond = get_bcc_types(db, cha, bond)
        # Seperate Correction for Aromatic Nitriles
        if (cha.TY[an] == 'CZ') and (set(['CA-CZ', 'CZ-NZ']) <= set(MOLBtype[an])):
            print(MOLBtype[an])
            print("Changing CZ-NZ to CZA-NZ")
            cha.loc[an, 'TY'] = 'CZA'
            bond = get_bcc_types(db, cha, bond)
        if (cha.TY[an] == 'CZ') and (set(['CT-CZ', 'CZ-NZ']) <= set(MOLBtype[an])):
            print(MOLBtype[an])
            print("Changing CZ-NZ to CZT-NZ")
            cha.loc[an, 'TY'] = 'CZT'
            bond = get_bcc_types(db, cha, bond)
        # Seperate Correction for 1,2,3 Amines
        if (cha.TY[an] == 'NT') and MOLBtype[an].count('H-NT') == 2:
            print(MOLBtype[an])
            print("Changing NT to NP")
            cha.loc[an, 'TY'] = 'NP'
            bond = get_bcc_types(db, cha, bond)
        if (cha.TY[an] == 'NT') and MOLBtype[an].count('H-NT') == 1:
            print("Changing NT to NS")
            cha.loc[an, 'TY'] = 'NS'
            bond = get_bcc_types(db, cha, bond)
        if (cha.TY[an] == 'NT') and MOLBtype[an].count('H-NT') == 0:
            print("Changing NT to N3")
            cha.loc[an, 'TY'] = 'N3'
            bond = get_bcc_types(db, cha, bond)
    cha = get_bcc_charges(db, bond, cha)
    QBCC = np.array(cha.QBCC)
    return (bond, cha, QBCC)


def get_bcc_charges(db, bond, cha):
    bond['IBCC'] = [sign * db[bcc] for sign, bcc in zip(bond.SI, bond.TIJ)]
    bond['JBCC'] = [sign * db[bcc] for sign, bcc in zip(bond.SJ, bond.TIJ)]
    cha['BCC'] = [
        sum(bond[bond['I'] == an + 1]['IBCC']) + sum(bond[bond['J'] == an + 1]['JBCC'])
        for an in cha.index
    ]
    cha['QBCC'] = cha['Q'] + cha['BCC']
    ars = [i for i in range(0, len(cha.TY)) if not cha['TY'][i].isdigit()]
    cha = cha.loc[ars]
    return cha
