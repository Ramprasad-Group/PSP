import psp.ChB_lib as bd
import numpy as np
import pandas as pd
import random
import math
from openbabel import openbabel as ob

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("xyz", "xyz")
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)


# define objective function
def f(
    unit_name,
    sl,
    unit,
    bond,
    angle,
    neigh_atoms_info,
    xyz_tmp_dir,
    dum1,
    dum2,
    atom1,
    atom2,
):
    file_name, conf_unit, dis_dum1_dum2, ang_1st_2nd, penalty = bd.create_conformer(
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
    )
    obConversion.ReadFile(mol, file_name)
    ff.Setup(mol)
    E_cost = (
        ff.Energy()
        + ff.Energy() * (1 - (ang_1st_2nd / 180.0))
        + ff.Energy() * penalty * 10
    )
    return E_cost, conf_unit, file_name


######################################################
# Simulated Annealing
######################################################
def SA(
    unit_name,
    unit,
    bonds,
    angle,
    neigh_atoms_info,
    xyz_tmp_dir,
    dum1,
    dum2,
    atom1,
    atom2,
    Steps,
    Substeps,
):
    i1 = bonds.index.values
    i2 = angle

    # Start location
    x_start = [i1[0], i2[0]]
    # Number of cycles
    n = Steps
    # Number of trials per cycle
    m = Substeps
    # Number of accepted solutions
    na = 0.0
    # Probability of accepting worse solution at the start
    p1 = 0.3
    # Probability of accepting worse solution at the end
    p50 = 0.001
    # Initial temperature
    t1 = -1.0 / math.log(p1)
    # Final temperature
    t50 = -1.0 / math.log(p50)

    # Fractional reduction every cycle
    frac = (t50 / t1) ** (1.0 / (n - 1.0))

    # Initialize x
    x = np.zeros((n + 1, 2))

    x[0] = x_start

    results = []

    xi = np.zeros(2)
    xi = x_start
    na = na + 1.0

    # Current best results so far
    xc = np.zeros(2)
    xc = x[0]
    fc, unit_new, file_name = f(
        unit_name,
        0,
        unit,
        bonds.loc[0],
        0.0,
        neigh_atoms_info,
        xyz_tmp_dir,
        dum1,
        dum2,
        atom1,
        atom2,
    )  # 102.5
    fs = np.zeros(n + 1)
    fs[0] = fc
    results.append([0, fc, file_name])

    # Current temperature
    t = t1
    # DeltaE Average
    DeltaE_avg = 0.0

    for i in range(n):
        for j in range(m):
            unit_prev = unit.copy()
            xi[0] = np.random.choice(i1)
            xi[1] = np.random.choice(i2)
            fc_new, unit, file_name = f(
                unit_name,
                i,
                unit,
                bonds.loc[xi[0]],
                xi[1],
                neigh_atoms_info,
                xyz_tmp_dir,
                dum1,
                dum2,
                atom1,
                atom2,
            )
            DeltaE = abs(fc_new - fc)

            if fc_new > fc:
                # Initialize DeltaE_avg if a worse solution was found
                #   on the first iteration
                if i == 0 and j == 0:
                    DeltaE_avg = DeltaE

                # To avoid divide by ZERO add a small number to DeltaE_avg
                if DeltaE_avg == 0.0:
                    DeltaE_avg = DeltaE_avg + 1.0e-13

                # objective function is worse
                # generate probability of acceptance
                p = math.exp(-DeltaE / (DeltaE_avg * t))

                # determine whether to accept worse point
                if random.random() < p:
                    # accept the worse solution
                    accept = True
                else:
                    # don't accept the worse solution
                    accept = False
            else:
                # objective function is lower, automatically accept
                accept = True

            if accept is True:
                # update currently accepted solution
                xc[0] = xi[0]
                xc[1] = xi[1]
                fc = fc_new
                best_xyz = file_name
                # increment number of accepted solutions
                na = na + 1.0
                # update DeltaE_avg
                DeltaE_avg = (DeltaE_avg * (na - 1.0) + DeltaE) / na

            else:
                unit = unit_prev.copy()

        # Record the best x values at the end of every cycle
        x[i + 1][0] = xc[0]
        x[i + 1][1] = xc[1]
        try:
            results.append([i, fc, best_xyz])
        except Exception:
            results.append([i, fc, 'XXX'])
        fs[i + 1] = fc

        if np.around(fs[i], decimals=15) == np.around(
            fs[i + 1], decimals=15
        ) and np.around(fs[i - 1], decimals=15) == np.around(fs[i + 1], decimals=15):
            break
        # Lower the temperature for next cycle
        t = frac * t
    results = pd.DataFrame(results, columns=['i', 'Energy+', 'xyzFile'])
    results = results[results['xyzFile'] != 'XXX']
    results = results.drop_duplicates(subset='xyzFile', keep="last")
    return results
