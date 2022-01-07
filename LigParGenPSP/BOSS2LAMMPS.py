"""
SCRIPT TO  WRITE LAMMPS DATA FILES LMP & INP Files
FROM BOSS ZMATRIX
Created on Mon Sep 30 03:31:05 2017
@author: Leela S. Dodda leela.dodda@yale.edu
@author: William L. Jorgensen Lab

REQUIREMENTS:
BOSS (need to set BOSSdir in bashrc and cshrc)
Preferably Anaconda python with following modules
pandas
argparse
numpy
"""

from LigParGenPSP.BOSSReader import bossPdbAtom2Element, ucomb
from LigParGenPSP.BOSSReader import bossElement2Mass, tor_cent
import pickle
import pandas as pd
import numpy as np


def Boss2LammpsLMP(resid, num2typ2symb, Qs, bnd_df, ang_df, tor_df, molecule_data):
    xyz_df = molecule_data.MolData["XYZ"]
    max_mol_size = 50
    prm = open(resid + ".lmp", "w+")
    prm.write("LAMMPS data file Created by - (Written by Leela S. Dodda)\n\n")
    prm.write("%8d atoms\n" % len(Qs))
    prm.write("%8d bonds\n" % len(bnd_df.KIJ))
    prm.write("%8d angles\n" % len(ang_df.K))
    prm.write("%8d dihedrals\n" % len(tor_df[tor_df.TY == "Proper"].index))
    prm.write("%8d impropers\n \n" % len(tor_df[tor_df.TY == "Improper"].index))
    prm.write("%8d atom types\n" % len(Qs))
    prm.write("%8d bond types\n" % len(bnd_df.KIJ))
    prm.write("%8d angle types\n" % len(ang_df.K))
    prm.write("%8d dihedral types\n" % len(tor_df[tor_df.TY == "Proper"].index))
    prm.write("%8d improper types\n \n" % len(tor_df[tor_df.TY == "Improper"].index))
    prm.write(
        "%12.6f %12.6f xlo xhi\n" % (xyz_df.X.min(), xyz_df.X.min() + max_mol_size)
    )
    prm.write(
        "%12.6f %12.6f ylo yhi\n" % (xyz_df.Y.min(), xyz_df.Y.min() + max_mol_size)
    )
    prm.write(
        "%12.6f %12.6f zlo zhi\n" % (xyz_df.Z.min(), xyz_df.Z.min() + max_mol_size)
    )
    # Printing Parameters for ALL BONDS/ANGLES/DIHEDRALS/IMPROPERS/Q/LJ #######
    prm.write("\nMasses\n\n")
    for i in range(len(Qs)):
        prm.write("%8d %10.3f  \n" % (i + 1, float(num2typ2symb[i][4])))
    prm.write("\nPair Coeffs \n\n")
    for i in range(len(Qs)):
        prm.write("%8d%11.3f%11.7f \n" % (i + 1, float(Qs[i][3]), float(Qs[i][2])))
    prm.write("\nBond Coeffs \n\n")
    for i in bnd_df.index:
        prm.write("%8d%11.4f%11.4f \n" % (i + 1, bnd_df.KIJ[i], bnd_df.RIJ[i]))
    prm.write("\nAngle Coeffs \n\n")
    for i in ang_df.index:
        prm.write("%8d%11.3f%11.3f\n" % (i + 1, ang_df.K[i], ang_df.R[i]))
    dihedral_df = tor_df[tor_df.TY == "Proper"]
    dihedral_df.index = range(len(dihedral_df.V1))
    prm.write("\nDihedral Coeffs  \n\n")
    for i, row in dihedral_df.iterrows():
        prm.write(
            "%8d%11.3f%11.3f%11.3f%11.3f \n" % (i + 1, row.V1, row.V2, row.V3, row.V4)
        )
    bndlist = list(bnd_df.UR) + (list(bnd_df.UR))
    improper_df = tor_df[tor_df.TY == "Improper"]
    improper_df.index = range(len(improper_df.V2))
    if len(improper_df.index) > 0:
        prm.write("\nImproper Coeffs \n\n")
        for i, row in improper_df.iterrows():
            prm.write("%8d%11.3f%8d%8d \n" % (i + 1, row.V2 * 0.5, -1, 2))
    # Printing EXPLICITLY ALL BONDS/ANGLES/DIHEDRALS/IMPROPERS/Q/LJ #######
    prm.write("\nAtoms \n\n")
    for i in range(len(xyz_df.index)):
        prm.write(
            "%6d %6d %6d %10.8f %8.3f %8.5f %8.5f\n"
            % (i + 1, 1, i + 1, float(Qs[i][1]), xyz_df.X[i], xyz_df.Y[i], xyz_df.Z[i])
        )
    prm.write("\nBonds \n\n")
    for i in bnd_df.index:
        prm.write(
            "%6d %6d %6d %6d\n" % (i + 1, i + 1, bnd_df.cl1[i] + 1, bnd_df.cl2[i] + 1)
        )
    prm.write("\nAngles \n\n")
    for i in ang_df.index:
        prm.write(
            "%6d %6d %6d %6d %6d\n"
            % (i + 1, i + 1, ang_df.cl1[i] + 1, ang_df.cl2[i] + 1, ang_df.cl3[i] + 1)
        )
    prm.write("\nDihedrals\n\n")
    for i, row in dihedral_df.iterrows():
        prm.write(
            "%6d %6d %6d %6d %6d %6d \n"
            % (i + 1, i + 1, row.I + 1, row.J + 1, row.K + 1, row.L + 1)
        )
    if len(improper_df.index) > 0:
        prm.write("\nImpropers\n\n")
        for row in improper_df.iterrows():
            index, dat = row
            ndata = tor_cent([dat.I, dat.J, dat.K, dat.L], bndlist)
            prm.write(
                "%6d %6d %6d %6d %6d %6d \n"
                % (
                    index + 1,
                    index + 1,
                    ndata[0] + 1,
                    ndata[1] + 1,
                    ndata[2] + 1,
                    ndata[3] + 1,
                )
            )
    return None


def Boss2CharmmTorsion(bnd_df, num2opls, st_no, molecule_data, num2typ2symb):
    dhd = []
    for line in molecule_data.MolData["TORSIONS"]:
        dt = [float(i) for i in line]
        dhd.append(dt)
    dhd = np.array(dhd)
    dhd = dhd  # kcal to kj conversion
    dhd = dhd  # Klammps = Vopls
    dhd_df = pd.DataFrame(dhd, columns=["V1", "V2", "V3", "V4"])
    ats = []
    for line in molecule_data.MolData["ATOMS"][3:]:
        dt = [line.split()[0], line.split()[4], line.split()[6], line.split()[8]]
        dt = [int(d) for d in dt]
        ats.append(dt)
    for line in molecule_data.MolData["ADD_DIHED"]:
        dt = [int(i) for i in line]
        ats.append(dt)
    assert len(ats) == len(
        dhd
    ), "Number of Dihedral angles in Zmatrix and Out file dont match"
    ats = np.array(ats) - st_no
    for i in range(len(ats)):
        for j in range(len(ats[0])):
            if ats[i][j] < 0:
                ats[i][j] = 0
    at_df = pd.DataFrame(ats, columns=["I", "J", "K", "L"])
    # final_df = pd.concat([dhd_df, at_df], axis=1, join_axes=[at_df.index]) backup
    final_df = pd.concat([dhd_df, at_df], axis=1)
    final_df = final_df.reindex(dhd_df.index)

    bndlist = list(bnd_df.UR) + (list(bnd_df.UR))
    final_df["TY"] = [
        "Proper"
        if ucomb(
            list([final_df.I[n], final_df.J[n], final_df.K[n], final_df.L[n]]), bndlist
        )
        == 3
        else "Improper"
        for n in range(len(final_df.I))
    ]
    final_df["TI"] = [num2typ2symb[j][2] for j in final_df.I]
    final_df["TJ"] = [num2typ2symb[j][2] for j in final_df.J]
    final_df["TK"] = [num2typ2symb[j][2] for j in final_df.K]
    final_df["TL"] = [num2typ2symb[j][2] for j in final_df.L]
    final_df["SYMB"] = [
        "-".join(
            [
                num2typ2symb[final_df.I[i]][0],
                num2typ2symb[final_df.J[i]][0],
                num2typ2symb[final_df.K[i]][0],
                num2typ2symb[final_df.L[i]][0],
            ]
        )
        for i in final_df.index
    ]
    if len(final_df.index) > 0:
        final_df["NAME"] = (
            final_df.TI + "-" + final_df.TJ + "-" + final_df.TK + "-" + final_df.TL
        )
    return final_df


def boss2CharmmBond(molecule_data, st_no):
    bdat = molecule_data.MolData["BONDS"]
    bdat["cl1"] = [x - st_no if not x - st_no < 0 else 0 for x in bdat["cl1"]]
    bdat["cl2"] = [x - st_no if not x - st_no < 0 else 0 for x in bdat["cl2"]]
    bnd_df = pd.DataFrame(bdat)
    bnd_df["UF"] = (
        (bnd_df.cl1 + bnd_df.cl2) * (bnd_df.cl1 + bnd_df.cl2 + 1) * 0.5
    ) + bnd_df.cl2
    bnd_df["UR"] = (
        (bnd_df.cl1 + bnd_df.cl2) * (bnd_df.cl1 + bnd_df.cl2 + 1) * 0.5
    ) + bnd_df.cl1
    hb_df = bnd_df.drop(["cl1", "cl2", "UF", "UR"], 1)
    hb_df = hb_df.drop_duplicates()
    return bnd_df


def boss2CharmmAngle(anglefile, num2opls, st_no):
    adat = anglefile
    adat["cl1"] = [x - st_no if not x - st_no < 0 else 0 for x in adat["cl1"]]
    adat["cl2"] = [x - st_no if not x - st_no < 0 else 0 for x in adat["cl2"]]
    adat["cl3"] = [x - st_no if not x - st_no < 0 else 0 for x in adat["cl3"]]
    ang_df = pd.DataFrame(adat)
    ang_df = ang_df[ang_df.K > 0]
    ang_df["TY"] = np.array(
        [
            num2opls[i] + "-" + num2opls[j] + "-" + num2opls[k]
            for i, j, k in zip(ang_df.cl1, ang_df.cl2, ang_df.cl3)
        ]
    )
    return ang_df


def bossData(molecule_data):
    ats_file = molecule_data.MolData["ATOMS"]
    types = []
    for i in enumerate(ats_file):
        types.append([i[1].split()[1], "opls_" + i[1].split()[2]])
    st_no = 3
    Qs = molecule_data.MolData["Q_LJ"]
    assert len(Qs) == len(types), "Please check the at_info and Q_LJ_dat files"
    num2opls = {}
    for i in range(0, len(types)):
        num2opls[i] = Qs[i][0]
    num2typ2symb = {i: types[i] for i in range(len(Qs))}
    for i in range(len(Qs)):
        num2typ2symb[i].append(
            bossPdbAtom2Element(num2typ2symb[i][0]) + num2typ2symb[i][1][-3:]
        )
        num2typ2symb[i].append(bossPdbAtom2Element(num2typ2symb[i][0]))
        num2typ2symb[i].append(bossElement2Mass(num2typ2symb[i][3]))
        num2typ2symb[i].append(Qs[i][0])
    return (types, Qs, num2opls, st_no, num2typ2symb)


def Boss2Lammps(resid, molecule_data):
    types, Qs, num2opls, st_no, num2typ2symb = bossData(molecule_data)
    bnd_df = boss2CharmmBond(molecule_data, st_no)
    ang_df = boss2CharmmAngle(molecule_data.MolData["ANGLES"], num2opls, st_no)
    tor_df = Boss2CharmmTorsion(bnd_df, num2opls, st_no, molecule_data, num2typ2symb)
    Boss2LammpsLMP(resid, num2typ2symb, Qs, bnd_df, ang_df, tor_df, molecule_data)
    return None


def mainBOSS2LAMMPS(resid, clu=False):
    mol = pickle.load(open(resid + ".p", "rb"))
    Boss2Lammps(resid, mol)
    return None
