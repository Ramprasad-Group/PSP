import numpy as np


def new_func(linex, match):
    out = 0
    for word in linex.split():
        if word == match:
            out = out + 1
    return out


def read_coords(data):
    cmatrix = []
    ta = []
    tb = []
    for i in range(0, len(data)):
        cmatrix.append(data[i].split())
    ta = [int(cmatrix[i][2]) for i in range(0, len(data))]
    tb = [int(cmatrix[i][3]) for i in range(0, len(data))]
    ta = np.array(ta)
    maxa = ta.max()
    tb = np.array(tb)
    numi = 1
    for i in range(0, len(tb)):
        if tb[i] > 1:
            tb[i] = maxa + numi
            numi = numi + 1
    for i in range(0, len(data)):
        cmatrix[i][3] = str(tb[i])
    outdat = []
    new_coord = ''
    for i in range(0, len(data)):
        new_coord = '{:>4s} {:<3s} {:>4s} {:>4s}'.format(
            cmatrix[i][0], cmatrix[i][1], cmatrix[i][2], cmatrix[i][3]
        )
        new_coord = new_coord + '{:>5s}{:>12s}{:>4s}{:>12s}'.format(
            cmatrix[i][4], cmatrix[i][5], cmatrix[i][6], cmatrix[i][7]
        )
        new_coord = new_coord + '{:>4s}{:>12s}{:>9s}'.format(
            cmatrix[i][8], cmatrix[i][9], cmatrix[i][10]
        )
        outdat.append(new_coord)
    tb = tb[tb > 0]  # IMPORTANT TO AVOID THE -1 and 0 IN FINAL ATOM TYPE
    ta = ta[ta > 0]  # IMPORTANT TO AVOID THE -1 and 0 IN FINAL ATOM TYPE
    return outdat, tb, ta


def read_files(infile):
    nline = 0
    cline = 0
    oline = 0
    data = []
    for line in infile:
        if line.rstrip():
            data.append(line)
            if "Non-Bonded" in line:
                oline = nline
            elif "Variations" in line:
                cline = nline
            nline += 1
    return data, nline, cline, oline


def rel_nbd(data, tb, QBCC=None):
    if QBCC is None:
        QBCC = np.zeros(len(data), dtype=float)
    nmat = []
    nmat = [ndat.split() for ndat in data]
    ondat = []
    for i in range(0, len(data)):
        nmat[i][0] = str(tb[i])
        nmat[i][3] = '%.6f' % QBCC[i]
        new_nb = '{:>4s}{:>3s} {:<3s} {:>9s} {:>9s} {:>9s}'.format(
            nmat[i][0], nmat[i][1], nmat[i][2], nmat[i][3], nmat[i][4], nmat[i][5]
        )
        ondat.append(new_nb)
    return ondat


def fepZmatFromFile(filenme, QBCC=None):
    qfile = open(filenme)
    qdat, nl1, cl1, ol1 = read_files(qfile)
    cdat, tb, ta = read_coords(qdat[1:cl1])
    ndat = rel_nbd(qdat[ol1 + 1:], tb, QBCC)
    qdat[ol1] = qdat[ol1].replace("AM1 CM1Ax1.14", "CM1Ax1.14TO1.14CM1A-BCC", 1)
    target = open(filenme[:-2] + '_fep.z', 'w')
    target.write(qdat[0])
    for i in range(0, len(cdat)):
        target.write(cdat[i] + '\n')
    for i in range(cl1, nl1):
        target.write(qdat[i])
    for i in range(0, len(ndat)):
        target.write(ndat[i] + '\n')
    target.close()
    return None


def fepZmatFromPkl(zmat_dat, filenme, QBCC=None):
    qdat, nl1, cl1, ol1 = read_files(zmat_dat)
    cdat, tb, ta = read_coords(qdat[1:cl1])
    ndat = rel_nbd(qdat[ol1 + 1:], tb, QBCC)
    qdat[ol1] = qdat[ol1].replace("AM1 CM1Ax1.14", "CM1Ax1.14TO1.14CM1A-BCC", 1)
    target = open(filenme + '_fep.z', 'w')
    target.write(qdat[0])
    for i in range(0, len(cdat)):
        target.write(cdat[i] + '\n')
    for i in range(cl1, nl1):
        target.write(qdat[i])
    for i in range(0, len(ndat)):
        target.write(ndat[i] + '\n')
    target.close()
    return None


def BCC_file2zmat(zmat, QBCC, oname):
    qfile = open(zmat, 'r+')
    qdat, nl1, cl1, ol1 = read_files(qfile)
    cdat, tb, ta = read_coords(qdat[1:cl1])
    ndat = rel_nbd(qdat[ol1 + 1:], ta, QBCC)
    qdat[ol1] = qdat[ol1].replace("AM1 CM1Ax1.14", "1.14CM1A-LBCC", 1)
    qfile.close()
    target = open(oname, 'w+')
    for i in range(0, ol1 + 1):
        target.write(qdat[i])
    for i in range(0, len(ndat)):
        target.write(ndat[i] + '\n')
    target.close()
    return None
