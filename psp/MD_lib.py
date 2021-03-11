import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.spatial.distance import cdist
from random import shuffle


def barycenter(unit):
    return unit.mean()


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


def add_mole(sys, unit):
    df = unit.copy()
    if sys.empty:
        df['i'] = 1
    else:
        df['i'] = max(list(sys.i.values)) + 1
    df['j'] = np.arange(1, len(unit.index) + 1, 1)
    sys = pd.concat([sys, df])
    return sys.reset_index(drop=True)


def get_initial_model(NMol_list, XYZ_list, tol_dis, xmin, xmax, ymin, ymax, zmin, zmax):
    # List index for all possible molecules
    all_mole_idx = []
    moleSN = 1
    for mole in NMol_list:
        all_mole_idx += [moleSN] * mole
        moleSN += 1
    # Shuffle indexes of molecules in the list
    shuffle(all_mole_idx)

    # create a system at origin
    per_incr = [0.0, 0.1, 0.2, 0.3]
    for per in per_incr:
        print("Percent increase: ", per)
        x_expand = (xmax - xmin) * per
        y_expand = (ymax - ymin) * per
        z_expand = (zmax - zmin) * per
        xmax_ex = xmax + x_expand
        ymax_ex = ymax + y_expand
        zmax_ex = zmax + z_expand

        # create a DataFrame for the system
        sys = pd.DataFrame()
        count = 0
        success = True
        add_yaxis = 0.0
        zlayer = 1

        for i in all_mole_idx:
            # print('>>>>>>>>>>', i)
            unit = pd.read_csv(
                XYZ_list[i - 1], header=None, skiprows=2, delim_whitespace=True
            )
            Natm = unit.shape[0]
            unit = move_barycenter(unit, 0, origin=True, barycenter=False)
            unit_mod, success, add_yaxis, zlayer = move_unit(
                unit,
                sys,
                tol_dis,
                xmin,
                xmax_ex,
                ymin,
                ymax_ex,
                zmin,
                zmax_ex,
                add_yaxis,
                zlayer=zlayer,
                Natm=Natm,
            )
            if success is True:
                count += 1
                sys = add_mole(sys, unit_mod)
            elif success is False and per < per_incr[-1]:
                break
            else:
                print("Can't pack molecules within the given box size.")
                exit()
        if success is True and per > 0.0:
            sys[1] = sys[1] - (sys[1].max() - xmax) / 2
            sys[2] = sys[2] - (sys[2].max() - ymax) / 2
            sys[3] = sys[3] - (sys[3].max() - zmax) / 2
            return sys
        elif success is True:
            return sys


def move_unit(
    unit,
    sys_mod,
    tol_dis,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    add_yaxis,
    zlayer=1,
    Natm=0,
):
    unit_mod = unit.copy()
    min_x_dis = unit_mod[1].max() - unit_mod[1].min() + tol_dis
    min_y_dis = unit_mod[2].max() - unit_mod[2].min() + tol_dis
    min_z_dis = unit_mod[3].max() - unit_mod[3].min() + tol_dis
    per = 0.0
    tol_dis_mod = tol_dis + per * tol_dis
    if sys_mod.empty is False:
        last_mol = sys_mod.tail(Natm)
        mol_xmax, mol_ymax, mol_zmax, = (
            last_mol[1].max(),
            last_mol[2].max(),
            last_mol[3].max(),
        )
        sys_xmax, sys_ymax, sys_zmax = (
            sys_mod[1].max(),
            sys_mod[2].max(),
            sys_mod[3].max(),
        )
        if (
            mol_zmax > zmax - min_z_dis
            and mol_ymax > ymax - min_y_dis
            and mol_xmax > xmax - min_x_dis
        ):
            # print("Can't pack molecules in the box; length of each axis is increased to 30%.")
            return unit_mod, False, add_yaxis, zlayer

        else:
            # print(sys_ymax, ymax - min_y_dis, mol_ymax)
            if mol_ymax > ymax - min_y_dis:  #
                # add_yaxis = 0.0
                # print(">>>>1")
                if mol_xmax > xmax - min_x_dis:
                    unit_mod[3] = unit_mod[3] + sys_mod[3].max() + tol_dis_mod
                    # print(">>>>2")
                    add_yaxis = unit_mod[2].max()
                    zlayer += 1
                    # print(unit_mod[2].max())
                else:
                    unit_mod[3] = unit_mod[3] + last_mol[3].min()
                    unit_mod[2] = unit_mod[2] + last_mol[2].min() - 0.1
                    unit_mod[1] = unit_mod[1] + last_mol[1].max() + tol_dis_mod

                    # print(">>>>3")
                    add_yaxis = max(add_yaxis, unit_mod[2].max())
                    # print(unit_mod[2].max())
            # elif sys_ymax > ymax-min_y_dis: #
            elif zlayer > 1:
                # print(">>>>4")
                if mol_xmax > xmax - min_x_dis:
                    if add_yaxis + min_y_dis < ymax:
                        # print(">>>>12")
                        unit_mod[3] = unit_mod[3] + last_mol[3].min()
                        # print('<<<<<<<<<< Do not enter >>>>>>>')
                        #    unit_mod[2] = unit_mod[2] + last_mol[2].max() + tol_dis_mod #+ 1.19### ISSUE
                        unit_mod[2] = unit_mod[2] + add_yaxis + tol_dis_mod
                    else:  # Add to z axis
                        # print(">>>>11")
                        unit_mod[3] = unit_mod[3] + sys_mod[3].max() + tol_dis_mod
                        zlayer += 1
                        # unit_mod[3] = unit_mod[3] + last_mol[3].min()
                        # unit_mod[2] = unit_mod[2] + add_yaxis + tol_dis_mod
                    add_yaxis = unit_mod[2].max()
                elif sys_xmax > xmax - min_x_dis:
                    unit_mod[3] = unit_mod[3] + last_mol[3].min()
                    unit_mod[2] = unit_mod[2] + last_mol[2].min()
                    unit_mod[1] = unit_mod[1] + last_mol[1].max() + tol_dis_mod
                    # print(">>>>6")
                    # print(unit_mod[2].max())
                    # add_yaxis = max(add_yaxis, unit_mod[2].max())
                else:
                    unit_mod[3] = unit_mod[3] + last_mol[3].min()
                    unit_mod[1] = unit_mod[1] + last_mol[1].max() + tol_dis_mod
                    # print(">>>>7")
                    # add_yaxis = max(add_yaxis, unit_mod[2].max())
                    # print(unit_mod[2].max())
                    # Additional conditions

            else:
                if mol_xmax > xmax - min_x_dis:
                    # unit_mod[2] = unit_mod[2] + last_mol[2].max() + tol_dis_mod
                    unit_mod[2] = unit_mod[2] + sys_mod[2].max() + tol_dis_mod
                    # New commands
                    # unit_mod[3] = unit_mod[3] + last_mol[3].min()
                    # unit_mod[1] = unit_mod[1] + last_mol[1].min()
                    # print(">>>>8")
                    # print(unit_mod[2].max())
                    add_yaxis = unit_mod[2].max()
                elif sys_xmax > xmax - min_x_dis:
                    # print(">>>>9", last_mol[2].min(), add_yaxis)
                    unit_mod[3] = unit_mod[3] + last_mol[3].min()
                    unit_mod[2] = unit_mod[2] + last_mol[2].min()
                    unit_mod[1] = unit_mod[1] + last_mol[1].max() + tol_dis_mod

                    add_yaxis = max(add_yaxis, unit_mod[2].max())
                    # print(unit_mod[2].max())
                else:
                    # print(">>>>10")
                    unit_mod[1] = unit_mod[1] + last_mol[1].max() + tol_dis_mod
                    add_yaxis = max(add_yaxis, unit_mod[2].max())

                    # add_yaxis = (max(add_yaxis, unit_mod[2].max()))
                    # print(unit_mod[2].max())
    return unit_mod, True, add_yaxis, zlayer


def get_vol(density, Nmol, molar_mass):
    return (Nmol * molar_mass * 10) / (6.02214076 * density)  # in Ang


def get_molar_mass(smi):
    return Descriptors.ExactMolWt(Chem.MolFromSmiles(smi))


def get_box_size(vol, box_type="cubic", incr_per=0.4):  # c = cubic; r = rectangular
    axis = vol ** (1.0 / 3.0)
    if box_type == 'r':
        zmax = axis + axis * incr_per
        axis2 = np.sqrt(vol / zmax)
        return 0, axis2, 0, axis2, 0, zmax
    else:
        return 0, axis, 0, axis, 0, axis


def evaluate_obj(sys, dis_cutoff, xmin, xmax, ymin, ymax, zmin, zmax):
    sys_dis = sys.copy()

    dis_value = 0
    list_mol = list(set(sys.i.values))
    # Remove last element
    list_mol.pop()

    for a in list_mol:
        unit1 = sys_dis[sys_dis['i'] == a]  # [[1,2,3]]
        unit2 = sys_dis[
            sys_dis['i'] != a
        ]  # [[1,2,3]] # need to improve. remove molecules far away from the unit1

        dist = cdist(unit1[[1, 2, 3]].values, unit2[[1, 2, 3]].values)
        new_arr = dist[
            dist < dis_cutoff
        ]  # If you may need to remove double counted distances (ij and ji)
        new_arr = dis_cutoff - new_arr

        dis_value = dis_value + np.sum(new_arr)
        sys_dis = sys_dis[sys_dis['i'] != a].reset_index(drop=True)

    bound_value = 0.0
    # X axis
    Arr_x = sys[1].values
    newArr_x_min = Arr_x[Arr_x < xmin]
    newArr_x_min = xmin - newArr_x_min

    newArr_x_max = Arr_x[Arr_x > xmax]
    newArr_x_max = newArr_x_max - xmax

    # Y axis
    Arr_y = sys[2].values
    newArr_y_min = Arr_y[Arr_y < ymin]
    newArr_y_min = ymin - newArr_y_min

    newArr_y_max = Arr_y[Arr_y > ymax]
    newArr_y_max = newArr_y_max - ymax

    # Z axis
    Arr_z = sys[3].values
    newArr_z_min = Arr_z[Arr_z < zmin]
    newArr_z_min = zmin - newArr_z_min

    newArr_z_max = Arr_z[Arr_z > zmax]
    newArr_z_max = newArr_z_max - zmax

    bound_value = (
        bound_value
        + np.sum(newArr_x_min)
        + np.sum(newArr_x_max)
        + np.sum(newArr_y_min)
        + np.sum(newArr_y_max)
        + np.sum(newArr_z_min)
        + np.sum(newArr_z_max)
    )

    return dis_value + bound_value


def rotateXYZ(unit, theta1, theta2, theta3):
    th1 = theta1 * np.pi / 180.0
    th2 = theta2 * np.pi / 180.0
    th3 = theta3 * np.pi / 180.0
    Rot_matrix = np.array(
        [
            [
                (np.cos(th1) * np.cos(th2) * np.cos(th3)) - (np.sin(th1) * np.sin(th3)),
                -(np.cos(th1) * np.cos(th2) * np.sin(th3))
                - (np.sin(th1) * np.cos(th3)),
                (np.cos(th1) * np.sin(th2)),
            ],
            [
                (np.sin(th1) * np.cos(th2) * np.cos(th3)) + (np.cos(th1) * np.sin(th3)),
                -(np.sin(th1) * np.cos(th2) * np.sin(th3))
                + (np.cos(th1) * np.cos(th3)),
                (np.sin(th1) * np.sin(th2)),
            ],
            [-(np.sin(th2) * np.cos(th3)), np.sin(th2) * np.sin(th3), np.cos(th2)],
        ]
    )

    rot_XYZ = unit.loc[:, [1, 2, 3]].copy()
    rotated_unit = rot_XYZ.values.dot(Rot_matrix)
    newXYZ = pd.DataFrame(rotated_unit, columns=[1, 2, 3])
    newXYZ.index = unit.index
    unit.loc[:, [1, 2, 3]] = newXYZ.loc[:, [1, 2, 3]]
    return unit


# This function generates a xyz file
# INPUT: Name of a output file and a DataFrame of element names and respective XYZ-coordinates
# OUTPUT: Write a XYZ file
def gen_sys_xyz(filename, unit):
    unit = unit[[0, 1, 2, 3]]
    with open(filename, 'w') as f:
        f.write(str(unit.values.shape[0]))  # NUMBER OF ATOMS
        f.write("\n\n")  # TWO SPACES
        unit.to_csv(
            f, sep=' ', index=False, header=False
        )  # XYZ COORDINATES OF NEW MOLECULE


def move_molecules(sys, disx, disy, disz, theta1, theta2, theta3):
    df = pd.DataFrame()
    for i in set(sys.i.values):
        Mi = sys[sys['i'] == i]
        Mi = move_barycenter(Mi, [disx[i - 1], disy[i - 1], disz[i - 1]], False)
        Mi = rotateXYZ(Mi, theta1[i - 1], theta2[i - 1], theta3[i - 1])
        df = pd.concat([df, Mi])
    return df


def gen_sys_vasp(filename, unit, xmin, xmax, ymin, ymax, zmin, zmax):
    unit = unit.sort_values(by=[0])
    add_dis = 0.4  # This additional distance (in Ang) is added to avoid interaction near boundary
    file = open(filename, 'w+')
    file.write('### ' + 'POSCAR' + ' ###\n')
    file.write('1\n')
    a_vec = xmax - xmin + add_dis
    b_vec = ymax - ymin + add_dis
    c_vec = zmax - zmin + add_dis

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


def gen_sys_data(
    filename, unit, xmin, xmax, ymin, ymax, zmin, zmax
):  # lammps data file
    unit = unit.sort_values(by=[0])
    # add_dis = 0.4 # This additional distance (in Ang) is added to avoid interaction near boundary
    file = open(filename, 'w+')
    file.write('### ' + '# LAMMPS data file written by PSP' + ' ###\n')
    file.write(str(unit.shape[0]) + ' atoms\n')
    file.write(str(len(list(set(unit[0].values)))) + ' atom types\n')
    file.write(str(xmin) + ' ' + str(xmax) + ' xlo xhi\n')
    file.write(str(ymin) + ' ' + str(ymax) + ' ylo yhi\n')
    file.write(str(zmin) + ' ' + str(zmax) + ' zlo zhi\n\n')

    ele_list = []
    ele_mass = []
    for element in sorted(set(unit[0].values)):
        ele_list.append(element)
        ele_mass.append(Chem.GetPeriodicTable().GetAtomicWeight(element))

    file.write('Masses\n\n')
    count = 1
    for mass in ele_mass:
        file.write(str(count) + ' ' + str(mass) + '\n')
        count += 1

    SN = np.arange(1, unit.shape[0] + 1)
    unit['SN'] = SN
    file.write('\nAtoms\n\n')
    file.write(unit[['SN', 1, 2, 3]].to_string(header=False, index=False))
    file.close()


def main_func(x, *args):
    arr_x = np.array_split(x, 6)
    disx = arr_x[0]
    disy = arr_x[1]
    disz = arr_x[2]
    theta1 = arr_x[3]
    theta2 = arr_x[4]
    theta3 = arr_x[5]
    sys = move_molecules(args[0], disx, disy, disz, theta1, theta2, theta3)
    return evaluate_obj(
        sys, args[1], args[2], args[3], args[4], args[5], args[6], args[7]
    )