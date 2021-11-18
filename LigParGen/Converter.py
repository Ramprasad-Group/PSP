from LigParGen.BOSSReader import BOSSReader, CheckForHs
from LigParGen.BOSS2LAMMPS import mainBOSS2LAMMPS
from LigParGen.CreatZmat import GenMolRep
import argparse
import pickle
import os
from openbabel import openbabel as ob

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("pdb", "mol")


def main():

    parser = argparse.ArgumentParser(
        prog='LigParGen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
    Ligand Parameter Generator Based on
    Jorgensen group's OPLS-AA/CM1A(-LBCC) FF
    Created on Mon Feb 15 15:40:05 2016
    @author: Leela S. Dodda leela.dodda@yale.edu
    @author: William L. Jorgensen Lab

    FF formats provided :
    --------------------
    OpenMM       .xml
    CHARMM/NAMD  .prm & .rtf
    GROMACS      .itp & .gro
    CNS/X-PLOR   .param & .top
    Q            .Q.prm & .Q.lib
    DESMOND      .cms
    BOSS/MCPRO   .z
    PDB2PQR      .pqr

    Input Files supported :
    --------------------
    SMILES code
    PDB
    MDL MOL Format

    ################################################
    if using MOL file
    Usage: LigParGen -m phenol.mol    -r PHN -c 0 -o 0

    if using PDB file
    Usage: LigParGen -p phenol.pdb    -r PHN -c 0 -o 0

    if using BOSS SMILES CODE
    Usage: LigParGen -s 'c1ccc(cc1)O' -r PHN -c 0 -o 0

    REQUIREMENTS:
    BOSS (need to set BOSSdir in bashrc and cshrc)
    Preferably Anaconda python with following modules
    pandas
    argparse
    numpy
    openbabel

    Please cite following references:
    1. LigParGen web server: an automatic OPLS-AA parameter generator for organic ligands
       Leela S. Dodda  Israel Cabeza de Vaca  Julian Tirado-Rives William L. Jorgensen
       Nucleic Acids Research, Volume 45, Issue W1, 3 July 2017, Pages W331–W336
    2. 1.14*CM1A-LBCC: Localized Bond-Charge Corrected CM1A Charges for Condensed-Phase Simulations
       Leela S. Dodda, Jonah Z. Vilseck, Julian Tirado-Rives , and William L. Jorgensen
       Department of Chemistry, Yale University, New Haven, Connecticut 06520-8107, United States
       J. Phys. Chem. B, 2017, 121 (15), pp 3864–3870
    3. Accuracy of free energies of hydration using CM1 and CM3 atomic charges.
       Udier–Blagović, M., Morales De Tirado, P., Pearlman, S. A. and Jorgensen, W. L.
       J. Comput. Chem., 2004, 25,1322–1332. doi:10.1002/jcc.20059
    """,
    )
    parser.add_argument("-r", "--resname", help="Residue name from PDB FILE", type=str)
    parser.add_argument(
        "-s", "--smiles", help="Paste SMILES code from CHEMSPIDER or PubChem", type=str
    )
    parser.add_argument(
        "-m", "--mol", help="Submit MOL file from CHEMSPIDER or PubChem", type=str
    )
    parser.add_argument(
        "-p", "--pdb", help="Submit PDB file from CHEMSPIDER or PubChem", type=str
    )
    parser.add_argument(
        "-o",
        "--opt",
        help="Optimization or Single Point Calculation",
        type=int,
        choices=[0, 1, 2, 3],
    )
    parser.add_argument(
        "-c",
        "--charge",
        type=int,
        choices=[0, -1, 1, -2, 2],
        help="0: Neutral <0: Anion >0: Cation ",
    )
    parser.add_argument(
        "-l",
        "--lbcc",
        help="Use 1.14*CM1A-LBCC charges instead of 1.14*CM1A",
        action="store_true",
    )
    parser.add_argument(
        "-d", "--outdir", help="PATH for output directory", type=str, default='.'
    )
    args = parser.parse_args()

    convert(**vars(args))


def convert(**kwargs):

    # set the default values
    options = {
        'opt': 0,
        'smiles': None,
        'zmat': None,
        'charge': 0,
        'lbcc': False,
        'mol': None,
        'resname': 'UNK',
        'pdb': None,
    }

    # update the default values based on the arguments
    options.update(kwargs)

    # set the arguments that you would used to get from argparse
    opt = options['opt']
    smiles = options['smiles']
    # zmat = options['zmat']
    charge = options['charge']
    lbcc = options['lbcc']
    resname = options['resname']
    mol = options['mol']
    pdb = options['pdb']
    outdir = options['outdir']
    if opt is not None:
        optim = opt
    else:
        optim = 0

    clu = False

    # assert (which('obabel')
    # is not None), "OpenBabel is Not installed or \n the executable location is not accessable"
    if os.path.exists(outdir + resname + '.xml'):
        os.system('/bin/rm ' + outdir + resname + '.*')
    if lbcc:
        if charge == 0:
            lbcc = True
            print('LBCC converter is activated')
        else:
            lbcc = False
            print(
                '1.14*CM1A-LBCC is only available for neutral molecules\n Assigning unscaled CM1A charges'
            )

    if smiles is not None:
        os.chdir(outdir)
        smifile = open('%s.smi' % resname, 'w+')
        smifile.write('%s' % smiles)
        smifile.close()
        GenMolRep('%s.smi' % resname, optim, resname, charge)
        mol = BOSSReader('%s.z' % resname, '%s' % outdir, optim, charge, lbcc)
    elif mol is not None:
        if not os.path.exists(os.path.join(outdir, mol)):
            os.system('cp %s %s' % (mol, outdir))
        os.chdir(outdir)
        GenMolRep(mol, optim, resname, charge)
        mol = BOSSReader('%s.z' % resname, '%s' % outdir, optim, charge, lbcc)
    elif pdb is not None:
        if not os.path.exists(os.path.join(outdir, pdb)):
            os.system('cp %s %s' % (pdb, outdir))
        os.chdir(outdir)
        # Convert pdb to mol using Obabelv3
        mole = ob.OBMol()
        obConversion.ReadFile(mole, pdb)
        mol = pdb.replace('pdb', 'mol')
        obConversion.WriteFile(mole, mol)
        GenMolRep(mol, optim, resname, charge)
        mol = BOSSReader('%s.z' % resname, '%s' % outdir, optim, charge, lbcc)
        clu = True
    assert (
        mol.MolData['TotalQ']['Reference-Solute'] == charge
    ), "PROPOSED CHARGE IS NOT POSSIBLE: SOLUTE MAY BE AN OPEN SHELL"
    assert CheckForHs(
        mol.MolData['ATOMS']
    ), "Hydrogens are not added. Please add Hydrogens"

    pickle.dump(mol, open(resname + ".p", "wb"))
    mainBOSS2LAMMPS(resname, clu)
    print('DONE WITH LAMMPS')

    # Cleanup
    list_files = [
        "sum",
        "log",
        "olog",
        "out",
        "optzmat",
        "slvzmat",
        "plt.pdb",
        "clu.pdb",
        "LL",
        "LBCC_BONDS.csv",
        "LBCC_CHARGES.csv ",
        resname + ".p",
        resname + ".z",
        resname + "_NO_LBCC.z",
    ]
    for file in list_files:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":

    main()
