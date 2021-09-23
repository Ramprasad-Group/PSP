import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')

# Get Idx for dummy atom
def get_linking_Idx(m,dum,iso_dum):
    isotope = int(iso_dum.replace(dum, ''))
    for atom in m.GetAtoms():
        if atom.GetSymbol() == dum and atom.GetIsotope() == isotope:
            if len([x.GetIdx() for x in atom.GetNeighbors()]) > 1:
                print("Check your SMILES")
                exit()
            return atom.GetIdx(), [x.GetIdx() for x in atom.GetNeighbors()][0]

def get3DfromRDKitmol(m,dir_xyz,unit_name):
    m2 = Chem.AddHs(m)
    AllChem.Compute2DCoords(m2)
    AllChem.EmbedMolecule(m2)
    AllChem.UFFOptimizeMolecule(m2, maxIters=200)
    Chem.MolToXYZFile(m2, dir_xyz + '/' +  unit_name + '.xyz')

def CombineTwoObmol(mol1,mol2,s1_Idx,s2_Idx, exist=False):
    builder = ob.OBBuilder()
    builder.SetKeepRings()

    # Remember the size of molecule
    mol1.DeleteAtom(mol1.GetAtom(s1_Idx[0]+1)) # OpenBabel counts from 1
      # OpenBabel counts from 1
    if exist is True:
        mol1.DeleteAtom(mol1.GetAtom(s2_Idx[0] + 1))
    else:
        mol2.DeleteAtom(mol2.GetAtom(s2_Idx[0]+1))

    if s1_Idx[0] < s1_Idx[1]:
        s1_Idx[1] = s1_Idx[1]-1
    if s2_Idx[0] < s2_Idx[1]:
        s2_Idx[1] = s2_Idx[1]-1

    print('mol1    ', s1_Idx[0]+1)
    print('mol2    ', s2_Idx[0]+1)
    #obConversion.WriteFile(mol1, out_xyz)
    n = mol1.NumAtoms()

    # Add second molecule
    if exist is False:
        mol1 += mol2
        builder.Connect(mol1, s1_Idx[1] + 1, s2_Idx[1] + n + 1)
    else:
        print("Connecting atoms :", s1_Idx[1] + 1, s2_Idx[1] + 1 -1)
        mol1.AddBond(s1_Idx[1] + 1, s2_Idx[1] + 1 -1, 1)

    #obConversion.WriteFile(mol1, out_xyz)

    # Optimize geometry
    ff.Setup(mol1)
    if exist is False:
        ff.ConjugateGradients(1000)
    else:
        ff.ConjugateGradients(100000)
    ff.SteepestDescent(1000)
    ff.UpdateCoordinates(mol1)
    return mol1, mol1, n
    #

