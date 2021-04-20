import numpy as np
import pandas as pd
import mmap
import os
import multiprocessing
import subprocess


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


def gen_bsse_xyz_coord(xyz_coord, len_a):
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


def check_output(file_name, cal_type):
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


def get_final_energy(file_name):
    for line in reversed(list(open(file_name))):
        matches = ["FINAL", "SINGLE", "POINT", "ENERGY"]
        if all(x in line for x in matches):
            return float(line.split()[4])


def get_bind_ener(cal_dir="", MolA_name="", MolB_name="", N_dimer=""):
    bind_ener_df = pd.DataFrame()
    path_file_a = cal_dir + "/" + MolA_name + "/" + MolA_name + ".out"
    check = check_output(path_file_a, "opt")
    if check == "pass":
        ener_a = get_final_energy(path_file_a)
    else:
        return bind_ener_df

    path_file_b = cal_dir + "/" + MolB_name + "/" + MolB_name + ".out"
    check = check_output(path_file_b, "opt")
    if check == "pass":
        ener_b = get_final_energy(path_file_b)
    else:
        return bind_ener_df

    bind_ener_list = []
    for i in range(1, N_dimer + 1):
        path_file_dimer = cal_dir + "/dimer/" + str(i) + "/" + str(i) + ".out"
        check = check_output(path_file_dimer, "opt")

        bsse_list = []
        if check == "pass":
            dimer_ener = get_final_energy(path_file_dimer)
            # BSSE
            # Order 1: A; 2: A + ghost orbital; 3: B; 4: B + ghost orbital
            for j in range(1, 5):
                path_file_bsse = (
                    cal_dir + "/dimer/" + str(i) + "/BSSE/" + str(j) + ".out"
                )
                check = check_output(path_file_bsse, "single")
                if check == "pass":
                    bsse_list.append(get_final_energy(path_file_bsse))
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
    check = check_output(output_name_MolA, "opt")
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
    check = check_output(output_name_MolB, "opt")
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
        check = check_output(output_name_dimer, "opt")
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
        check = check_output(output_name_dimer, "opt")
        if check == "pass":
            opt_xyz_name_dimer = (
                cal_dir + "/dimer" + "/" + str(i) + "/" + str(i) + ".xyz"
            )
            xyz_coord_opt_dimer = pd.read_csv(
                opt_xyz_name_dimer, header=None, skiprows=2, delim_whitespace=True
            )
            # xyz_coord for bsse a1: MolA; a2: molA + gh Orb; b3: MolB; b4: MolB + gh Orb
            a1, a2, b3, b4 = gen_bsse_xyz_coord(
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
                check = check_output(output_name_orca_bsse, "single")
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
