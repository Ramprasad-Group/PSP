import math
import numpy as np


def bossElement2Num(elem):
    symb2mass = {
        "H": 1,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Br": 35,
        "I": 53,
    }
    try:
        res = symb2mass[elem]
    except NameError:
        print(
            "Mass for atom %s is not available \n add it to symb2mass dictionary"
        )
    return res


def pairing_func(a, b):
    ans = (a + b) * (a + b + 1) * 0.5
    if a > b:
        ans = ans + a
    else:
        ans = ans + b
    return int(ans)


def Vector(x, y, z):
    return (x, y, z)


def length(v):
    "Return length of a vector."
    sum = 0.0
    for c in v:
        sum += c * c
    return math.sqrt(sum)


def subtract(u, v):
    "Return difference between two vectors."
    x = u[0] - v[0]
    y = u[1] - v[1]
    z = u[2] - v[2]
    return Vector(x, y, z)


def dot(u, v):
    "Return dot product of two vectors."
    sum = 0.0
    for cu, cv in zip(u, v):
        sum += cu * cv
    return sum


def Distance(u, v):
    "Return length of a vector."
    #    print(u,v)
    uv = subtract(u, v)
    lsum = 0.0
    for c in uv:
        lsum += c * c
    return math.sqrt(lsum)


def cross(u, v):
    "Return the cross product of two vectors."
    x = u[1] * v[2] - u[2] * v[1]
    y = u[2] * v[0] - u[0] * v[2]
    z = u[0] * v[1] - u[1] * v[0]
    return Vector(x, y, z)


def Mol_angle(v0, v1):
    "Return angle [0..pi] between two vectors."
    cosa = round(dot(v0, v1) / length(v0) / length(v1), 3)
    return np.arccos(cosa)


def angle(p0, p1, p2):
    "Return angle [0..pi] between two vectors."
    v0 = subtract(p0, p1)
    v1 = subtract(p2, p1)
    cosa = dot(v0, v1) / length(v0) / length(v1)
    #    print(cosa)
    return 180.0 * np.arccos(round(cosa, 3)) * 7.0 / 22.0


def dihedral(p0, p1, p2, p3):
    "Return angle [0..2*pi] formed by vertices p0-p1-p2-p3."
    v01 = subtract(p0, p1)
    v32 = subtract(p3, p2)
    v12 = subtract(p1, p2)
    v0 = cross(v12, v01)
    v3 = cross(v12, v32)
    # The cross product vectors are both normal to the axis
    # vector v12, so the angle between them is the dihedral
    # angle that we are looking for.  However, since "angle"
    # only returns values between 0 and pi, we need to make
    # sure we get the right sign relative to the rotation axis
    a = Mol_angle(v0, v3)
    if dot(cross(v0, v3), v12) > 0:
        a = -a
    return a * 180.0 * 7.0 / 22.0


def tor_id(a):
    bond = pairing_func(a[1], a[2])
    ends = pairing_func(a[0], a[3])
    return "%d-%d" % (bond, ends)


def ang_id(a):
    bond_a = pairing_func(a[0], a[1])
    bond_b = pairing_func(a[1], a[2])
    return pairing_func(bond_a, bond_b)
