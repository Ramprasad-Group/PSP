import numpy as np
import pandas as pd
import mmap
import os
import multiprocessing
import subprocess
from rdkit import Chem


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


# This function try to create a directory
def build_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print(" Directory " + path + " already exists. ")
        pass
        # exit()


def run_orca(bashCommand, output, openmpi_path, openmpi_lib_path):
    f = open(output, "w")
    process = subprocess.Popen(
        openmpi_lib_path + ";" + openmpi_path + ";" + bashCommand, stdout=f, shell=True
    )  # stdout=subprocess.PIPE

    output, error = process.communicate()
    return error


def gen_orca_inp(
    filename,
    unit,
    func="B3LYP/G",
    basis="6-31G(d,p)",
    opt="",
    extra_key="TightSCF grid5 NoFinalGrid ANGS",
    charg=0,
    mult=1,
    nproc=multiprocessing.cpu_count(),
    mem=5000,
    max_iter=1000,
    Geo_const=False,
    atm_const=[],
):
    with open(filename, "w") as f:
        f.write("!" + func + " " + basis + " " + extra_key + " " + opt + "\n\n")
        f.write("%pal nprocs " + str(nproc) + "\nend\n\n")
        if Geo_const is True:
            f.write("%geom Constraints\n")
            for atm in atm_const:  # List of atoms, Always count atoms from 0
                f.write("    {C " + atm + " C}\n")
            f.write("\nend\nend")
        if len(opt) != 0:
            f.write("%geom MaxIter " + str(max_iter) + "\nend\n\n")
        f.write("%MaxCore " + str(mem) + "\n\n")
        f.write("*xyz " + str(charg) + " " + str(mult) + "\n")
        unit.to_csv(
            f, sep=" ", index=False, header=False
        )  # XYZ COORDINATES OF NEW MOLECULE
        f.write("*")


def orca_cal(
    cal_dir="",
    input_xyz="",
    MolA_name="",
    MolB_name="",
    N_dimer="",
    functional="",
    basis_set="",
    charge="",
    unpaired_elec="",
    orca_extra_keywords="",
    NCore="",
):
    # Create directory
    build_dir(cal_dir)
    build_dir(cal_dir + "/" + MolA_name)
    build_dir(cal_dir + "/" + MolB_name)
    build_dir(cal_dir + "/dimer")

    # Orca and OpenMPI paths
    orca_path = os.getenv("ORCA_EXEC")
    openmpi_path = "export PATH=" + os.getenv("OPENMPI_bin") + ":$PATH"
    openmpi_lib_path = (
        "export LD_LIBRARY_PATH=" + os.getenv("OPENMPI_lib") + ":$LD_LIBRARY_PATH"
    )

    # Read xyz coordinates and create orca inputs in respective directories
    # Geometry optimization of molecule A
    xyz_coord_MolA = pd.read_csv(
        input_xyz + "/" + str(MolA_name) + ".xyz",
        header=None,
        skiprows=2,
        delim_whitespace=True,
    )
    output_name_MolA = cal_dir + "/" + str(MolA_name) + "/" + str(MolA_name) + ".out"
    check = check_output_orca(output_name_MolA, "opt")
    if check != "pass":
        input_name_MolA = cal_dir + "/" + str(MolA_name) + "/" + str(MolA_name) + ".inp"
        gen_orca_inp(
            input_name_MolA,
            xyz_coord_MolA,
            opt="OPT",
            func=functional,
            basis=basis_set,
            charg=charge[0],
            mult=int(2 * (unpaired_elec[0]) * (1 / 2) + 1),
            extra_key=orca_extra_keywords,
            nproc=NCore,
        )

        # Orca calculations
        command = orca_path + " " + input_name_MolA
        err_MolA = run_orca(command, output_name_MolA, openmpi_path, openmpi_lib_path)

    # Geometry optimization of molecule B
    output_name_MolB = cal_dir + "/" + str(MolB_name) + "/" + str(MolB_name) + ".out"
    check = check_output_orca(output_name_MolB, "opt")
    if check != "pass":
        input_name_MolB = cal_dir + "/" + str(MolB_name) + "/" + str(MolB_name) + ".inp"
        xyz_coord_MolB = pd.read_csv(
            input_xyz + "/" + str(MolB_name) + ".xyz",
            header=None,
            skiprows=2,
            delim_whitespace=True,
        )
        gen_orca_inp(
            input_name_MolB,
            xyz_coord_MolB,
            opt="OPT",
            func=functional,
            basis=basis_set,
            charg=charge[1],
            mult=int(2 * (unpaired_elec[1]) * (1 / 2) + 1),
            extra_key=orca_extra_keywords,
            nproc=NCore,
        )

        # Orca calculations
        command = orca_path + " " + input_name_MolB
        err_MolB = run_orca(command, output_name_MolB, openmpi_path, openmpi_lib_path)

    # Geometry optimizations of dimers
    for i in range(1, N_dimer + 1):
        output_name_dimer = cal_dir + "/dimer" + "/" + str(i) + "/" + str(i) + ".out"
        check = check_output_orca(output_name_dimer, "opt")
        if check != "pass":
            input_name_dimer = cal_dir + "/dimer" + "/" + str(i) + "/" + str(i) + ".inp"
            build_dir(cal_dir + "/dimer" + "/" + str(i))
            xyz_coord_dimer = pd.read_csv(
                input_xyz + "/" + str(i) + ".xyz",
                header=None,
                skiprows=2,
                delim_whitespace=True,
            )
            gen_orca_inp(
                input_name_dimer,
                xyz_coord_dimer,
                opt="OPT",
                func=functional,
                basis=basis_set,
                charg=charge[0] + charge[1],
                mult=int(2 * (unpaired_elec[0] + unpaired_elec[1]) * (1 / 2) + 1),
                extra_key=orca_extra_keywords,
                nproc=NCore,
            )

            # Orca calculations
            command = orca_path + " " + input_name_dimer
            err_dimer = run_orca(
                command, output_name_dimer, openmpi_path, openmpi_lib_path
            )

    # BSSE calculation
    for i in range(1, N_dimer + 1):
        output_name_dimer = cal_dir + "/dimer" + "/" + str(i) + "/" + str(i) + ".out"
        check = check_output_orca(output_name_dimer, "opt")
        if check == "pass":
            opt_xyz_name_dimer = (
                cal_dir + "/dimer" + "/" + str(i) + "/" + str(i) + ".xyz"
            )
            xyz_coord_opt_dimer = pd.read_csv(
                opt_xyz_name_dimer, header=None, skiprows=2, delim_whitespace=True
            )
            # xyz_coord for bsse a1: MolA; a2: molA + gh Orb; b3: MolB; b4: MolB + gh Orb
            a1, a2, b3, b4 = gen_bsse_xyz_coord_orca(
                xyz_coord_opt_dimer, xyz_coord_MolA.shape[0]
            )
            xyz_coord_bsse = [a1, a2, b3, b4]
            # Create BSSE directory
            build_dir(cal_dir + "/dimer" + "/" + str(i) + "/" + "BSSE")
            for j in range(1, 5):
                output_name_orca_bsse = (
                    cal_dir
                    + "/dimer"
                    + "/"
                    + str(i)
                    + "/"
                    + "BSSE"
                    + "/"
                    + str(j)
                    + ".out"
                )
                check = check_output_orca(output_name_orca_bsse, "single")
                if check != "pass":
                    input_name_orca_bsse = (
                        cal_dir
                        + "/dimer"
                        + "/"
                        + str(i)
                        + "/"
                        + "BSSE"
                        + "/"
                        + str(j)
                        + ".inp"
                    )
                    if j <= 2:  # First molecule
                        charg_bsse = charge[0]
                        multi_bsse = int(2 * (unpaired_elec[0]) * (1 / 2) + 1)
                    else:  # Second molecule
                        charg_bsse = charge[1]
                        multi_bsse = int(2 * (unpaired_elec[1]) * (1 / 2) + 1)
                    gen_orca_inp(
                        input_name_orca_bsse,
                        xyz_coord_bsse[j - 1],
                        func=functional,
                        basis=basis_set,
                        charg=charg_bsse,
                        mult=multi_bsse,
                        extra_key=orca_extra_keywords,
                        nproc=NCore,
                    )

                    # Orca calculations
                    command = orca_path + " " + input_name_orca_bsse
                    err_dimer = run_orca(
                        command, output_name_orca_bsse, openmpi_path, openmpi_lib_path
                    )


def check_output_orca(file_name, cal_type):
    if os.path.isfile(file_name) is False:
        return "fail"
    with open(file_name, "rb", 0) as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as s:
        if s.find(b"***ORCA TERMINATED NORMALLY****") != -1:
            if cal_type == "opt":
                if s.find(b"***        THE OPTIMIZATION HAS CONVERGED     ***") != -1:
                    return "pass"
                else:
                    print(file_name, "*** The optimization is not CONVERGED ***")
                    return "fail"
            else:
                return "pass"
        else:
            print(file_name, "*** Error in output file ***")
            return "fail"


def gen_bsse_xyz_coord_orca(xyz_coord, len_a):
    a1 = xyz_coord[:len_a]
    a1_ = a1.copy()
    a1_[4] = ":"

    b3 = xyz_coord[len_a:]
    b3_ = b3.copy()
    b3_[4] = ":"

    a2 = pd.concat([a1, b3_])
    a2 = a2.replace(np.nan, "", regex=True)
    a2 = a2[[0, 4, 1, 2, 3]]

    b4 = pd.concat([a1_, b3])
    b4 = b4.replace(np.nan, "", regex=True)
    b4 = b4[[0, 4, 1, 2, 3]]
    return a1, a2, b3, b4


def get_final_energy_orca(file_name):
    for line in reversed(list(open(file_name))):
        matches = ["FINAL", "SINGLE", "POINT", "ENERGY"]
        if all(x in line for x in matches):
            return float(line.split()[4])


def get_bind_ener_orca(cal_dir="", MolA_name="", MolB_name="", N_dimer=""):
    bind_ener_df = pd.DataFrame()
    path_file_a = cal_dir + "/" + MolA_name + "/" + MolA_name + ".out"
    check = check_output_orca(path_file_a, "opt")
    if check == "pass":
        ener_a = get_final_energy_orca(path_file_a)
    else:
        return bind_ener_df

    path_file_b = cal_dir + "/" + MolB_name + "/" + MolB_name + ".out"
    check = check_output_orca(path_file_b, "opt")
    if check == "pass":
        ener_b = get_final_energy_orca(path_file_b)
    else:
        return bind_ener_df

    bind_ener_list = []
    for i in range(1, N_dimer + 1):
        path_file_dimer = cal_dir + "/dimer/" + str(i) + "/" + str(i) + ".out"
        check = check_output_orca(path_file_dimer, "opt")

        bsse_list = []
        if check == "pass":
            dimer_ener = get_final_energy_orca(path_file_dimer)
            # BSSE
            # Order 1: A; 2: A + ghost orbital; 3: B; 4: B + ghost orbital
            for j in range(1, 5):
                path_file_bsse = (
                    cal_dir + "/dimer/" + str(i) + "/BSSE/" + str(j) + ".out"
                )
                check = check_output_orca(path_file_bsse, "single")
                if check == "pass":
                    bsse_list.append(get_final_energy_orca(path_file_bsse))
                else:
                    break
            if check == "pass":
                bsse_corr = bsse_list[1] - bsse_list[0] + bsse_list[3] - bsse_list[2]
                bind_ener = dimer_ener - ener_a - ener_b - bsse_corr
                bind_ener_list.append([i, bind_ener * 627.503])
        # else:
        # break
    bind_ener_df = pd.DataFrame(bind_ener_list, columns=["SN", "BE (kcal/mol)"])
    return bind_ener_df


def run_gamess(bashCommand, scrach_dir, input_file_name):
    f = open(input_file_name.split('.')[0] + ".out", "w")
    process = subprocess.Popen(
        bashCommand + " " + input_file_name.split('.')[0], stdout=f, shell=True
    )  # stdout=subprocess.PIPE
    output, error = process.communicate()
    os.remove(
        os.path.join(scrach_dir, input_file_name.split('.')[0].split('/')[-1] + ".dat")
    )
    return error


def gen_gamess_input(filename, unit, keywords, XYZ=False):
    if XYZ is True:
        unit.columns = ["ATOM", "X", "Y", "Z"]

        ZNUC = []
        for index, row in unit.iterrows():
            ZNUC.append(float(Chem.rdchem.Atom(row[0]).GetAtomicNum()))
        unit['CHARGE'] = ZNUC
    with open(filename, "w") as f:
        f.write("! File created by PSP\n")
        for key in keywords:
            f.write(key + "\n")
        f.write("\n $DATA\nTitle\nC1\n")
        unit[["ATOM", "CHARGE", "X", "Y", "Z"]].to_csv(
            f, sep=" ", index=False, header=False
        )
        f.write(" $END")


def gamess_cal(
    cal_dir="",
    input_xyz="",
    MolA_name="",
    MolB_name="",
    N_dimer="",
    gamess_path="",
    scrach_dir="",
    keywords_dict="",
):
    # Create directory
    build_dir(cal_dir)
    MolA_dir = os.path.join(cal_dir, MolA_name)
    MolB_dir = os.path.join(cal_dir, MolB_name)
    dimer_dir = os.path.join(cal_dir, "dimer")

    build_dir(MolA_dir)
    build_dir(MolB_dir)
    build_dir(dimer_dir)

    # Read xyz coordinates and create gamess inputs in respective directories
    # Geometry optimization of molecule A
    xyz_coord_MolA = pd.read_csv(
        os.path.join(input_xyz, str(MolA_name) + ".xyz"),
        header=None,
        skiprows=2,
        delim_whitespace=True,
    )
    output_name_MolA = os.path.join(MolA_dir, str(MolA_name) + ".out")
    check = check_output_gamess(output_name_MolA, "OPT")

    if check != "pass":
        input_name_MolA = os.path.join(MolA_dir, str(MolA_name) + ".inp")
        gen_gamess_input(
            input_name_MolA, xyz_coord_MolA, keywords_dict["MolA_opt"], XYZ=True
        )

        # gamess calculations
        run_gamess(gamess_path, scrach_dir, input_name_MolA)

    # Geometry optimization of molecule B
    xyz_coord_MolB = pd.read_csv(
        os.path.join(input_xyz, str(MolB_name) + ".xyz"),
        header=None,
        skiprows=2,
        delim_whitespace=True,
    )
    output_name_MolB = os.path.join(MolB_dir, str(MolB_name) + ".out")
    check = check_output_gamess(output_name_MolB, "OPT")

    if check != "pass":
        input_name_MolB = os.path.join(MolB_dir, str(MolB_name) + ".inp")
        gen_gamess_input(
            input_name_MolB, xyz_coord_MolB, keywords_dict["MolB_opt"], XYZ=True
        )

        # gamess calculations
        run_gamess(gamess_path, scrach_dir, input_name_MolB)

    # Geometry optimizations of dimers
    for i in range(1, N_dimer + 1):
        output_name_dimer = os.path.join(dimer_dir, str(i) + "/" + str(i) + ".out")
        check = check_output_gamess(output_name_dimer, "OPT")
        if check != "pass":
            dimer_dir_each = os.path.join(dimer_dir, str(i))
            input_name_dimer = os.path.join(dimer_dir_each, str(i) + ".inp")
            build_dir(dimer_dir_each)
            xyz_coord_dimer = pd.read_csv(
                os.path.join(input_xyz, str(i) + ".xyz"),
                header=None,
                skiprows=2,
                delim_whitespace=True,
            )
            gen_gamess_input(
                input_name_dimer, xyz_coord_dimer, keywords_dict["MolAB_opt"], XYZ=True
            )

            # gamess calculations
            run_gamess(gamess_path, scrach_dir, input_name_dimer)

    # BSSE calculation
    for i in range(1, N_dimer + 1):
        output_name_dimer = os.path.join(dimer_dir, str(i) + "/" + str(i) + ".out")
        check = check_output_gamess(output_name_dimer, "OPT")
        if check == "pass":
            Zxyz_coord_opt_dimer = get_opt_xyz_gamess(output_name_dimer)
            # xyz_coord for bsse a1: MolA; a2: molA + gh Orb; b3: MolB; b4: MolB + gh Orb
            a1, a2, b3, b4 = gen_bsse_xyz_coord_gamess(
                Zxyz_coord_opt_dimer, xyz_coord_MolA.shape[0]
            )
            xyz_coord_bsse = [a1, a2, b3, b4]
            # Create BSSE directory
            bsse_dir = os.path.join(dimer_dir, str(i) + "/" + "BSSE")
            build_dir(bsse_dir)

            for j in range(1, 5):
                output_name_orca_bsse = os.path.join(bsse_dir, str(j) + ".out")
                check = check_output_gamess(output_name_orca_bsse)
                if check != "pass":
                    input_name_orca_bsse = os.path.join(bsse_dir, str(j) + ".inp")
                    if j <= 2:
                        gen_gamess_input(
                            input_name_orca_bsse,
                            xyz_coord_bsse[j - 1],
                            keywords_dict["MolA_sp"],
                        )
                    else:
                        gen_gamess_input(
                            input_name_orca_bsse,
                            xyz_coord_bsse[j - 1],
                            keywords_dict["MolB_sp"],
                        )
                    # gamess calculations
                    run_gamess(gamess_path, scrach_dir, input_name_orca_bsse)


def check_output_gamess(file_name, cal_type="ENER"):
    if os.path.isfile(file_name) is False:
        return "fail"
    with open(file_name, "rb", 0) as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as s:
        if s.find(b"EXECUTION OF GAMESS TERMINATED NORMALLY") != -1:
            if cal_type == "OPT":
                if s.find(b"***** EQUILIBRIUM GEOMETRY LOCATED *****") != -1:
                    return "pass"
                else:
                    print(file_name, "*** The optimization is not CONVERGED ***")
                    return "fail"
            else:
                return "pass"
        else:
            print(file_name, "*** Error in output file ***")
            return "fail"


def get_opt_xyz_gamess(filename):
    data = []
    flag = False
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('***** EQUILIBRIUM GEOMETRY LOCATED *****'):
                flag = True
            elif line.strip().endswith('INTERNUCLEAR DISTANCES (ANGS.)'):
                flag = False
            elif flag:
                list_ = line.strip().split()
                if "COORDINATES" not in list_:
                    data.append(list_)
    # data = pd.DataFrame(data[3:-1])
    data = pd.DataFrame(data).dropna()
    data.columns = data.iloc[0].values
    data = data.iloc[1:].reset_index(drop=True)
    return data


def get_xyz_gamess(filename):
    flag = False
    data_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('BEGINNING GEOMETRY SEARCH POINT NSERCH'):
                data = []
                SN = int(line.split()[5])
                flag = True
            elif flag is True and line.strip().endswith(
                'INTERNUCLEAR DISTANCES (ANGS.)'
            ):
                data = pd.DataFrame(data).dropna()
                data.columns = data.iloc[0].values
                data = data.iloc[1:].reset_index(drop=True)
                data_dict[SN] = data
                flag = False
            elif flag is True and line.strip().endswith('********************'):
                data = pd.DataFrame(data).dropna()
                data.columns = data.iloc[0].values
                data = data.iloc[1:].reset_index(drop=True)
                data_dict[SN] = data
                flag = False
            elif flag:
                # print(line)
                list_ = line.strip().split()
                if "COORDINATES" not in list_:
                    data.append(list_)
    # data = pd.DataFrame(data[3:-1])
    return data_dict


def gen_bsse_xyz_coord_gamess(xyz_coord, len_a):
    a1 = xyz_coord[:len_a]
    a1_ = a1.copy()
    a1_['CHARGE'] = '-' + a1_['CHARGE']

    b3 = xyz_coord[len_a:]
    b3_ = b3.copy()
    b3_['CHARGE'] = '-' + b3_['CHARGE']

    a2 = pd.concat([a1, b3_])

    b4 = pd.concat([a1_, b3])
    return a1, a2, b3.reset_index(drop=True), b4


def get_final_energy_gamess(file_name):
    for line in reversed(list(open(file_name))):
        matches = ["TOTAL ENERGY"]
        if all(x in line for x in matches):
            return float(line.split()[3])


def get_energy_gamess(file_name):
    ener = []
    for line in list(open(file_name)):
        matches = ["GRAD. MAX="]
        if all(x in line for x in matches):
            ener.append([line.split()[index] for index in [1, 3]])
            # ener.append(line.split()[1,3])
    ener = pd.DataFrame(ener, columns=[["SN", "ENER"]])
    return ener
    #    return float(line.split()[3])


def get_bind_ener_gamess(cal_dir="", MolA_name="", MolB_name="", N_dimer=""):
    bind_ener_df = pd.DataFrame()
    path_file_a = os.path.join(cal_dir, MolA_name + "/" + MolA_name + ".out")
    check = check_output_gamess(path_file_a, "OPT")
    if check == "pass":
        ener_a = get_final_energy_gamess(path_file_a)
    else:
        return bind_ener_df

    path_file_b = os.path.join(cal_dir, MolB_name + "/" + MolB_name + ".out")
    check = check_output_gamess(path_file_b, "OPT")
    if check == "pass":
        ener_b = get_final_energy_gamess(path_file_b)
    else:
        return bind_ener_df

    bind_ener_list = []
    for i in range(1, N_dimer + 1):
        path_file_dimer = os.path.join(
            cal_dir, "dimer/" + str(i) + "/" + str(i) + ".out"
        )
        check = check_output_gamess(path_file_dimer, "OPT")

        bsse_list = []
        if check == "pass":
            dimer_ener = get_final_energy_gamess(path_file_dimer)
            # BSSE
            # Order 1: A; 2: A + ghost orbital; 3: B; 4: B + ghost orbital
            for j in range(1, 5):
                path_file_bsse = os.path.join(
                    cal_dir, "dimer/" + str(i) + "/BSSE/" + str(j) + ".out"
                )
                check = check_output_gamess(path_file_bsse)
                if check == "pass":
                    bsse_list.append(get_final_energy_gamess(path_file_bsse))
                else:
                    break
            if check == "pass":
                bsse_corr = bsse_list[1] - bsse_list[0] + bsse_list[3] - bsse_list[2]
                bind_ener = dimer_ener - ener_a - ener_b - bsse_corr
                bind_ener_list.append([i, bind_ener * 627.503])
        # else:
        # break
    bind_ener_df = pd.DataFrame(bind_ener_list, columns=["SN", "BE (kcal/mol)"])
    return bind_ener_df
