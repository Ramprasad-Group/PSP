import numpy as np
import pandas as pd
import psp.BE_lib as BElib
import mmap
import io
import re
# PATHS and other inputs
orca_path = '/home/hari/.soft/orca_4_2_OMPI316/orca'
openmpi_lib_path = 'export LD_LIBRARY_PATH=/home/hari/.soft/openmpi-316/lib:$LD_LIBRARY_PATH'
openmpi_path = 'export PATH=/home/hari/.soft/openmpi-316/bin:$PATH'
input_xyz = '/home/hari/project/usa/psp_v1/PSP/v10/PSP/test/DimerBuilder/dimer-xyz'
MolA_name = 'A'
MolB_name = 'B'
cal_dir = '/home/hari/project/usa/psp_v1/PSP/v10/PSP/test/DimerBuilder/dft_calc_orca'
N_dimer = 10
NCore = 8
functional = "B3LYP/G"
basis_set = "6-31G(d,p)"
orca_extra_keywords = "D3BJ TightSCF grid5 NoFinalGrid ANGS"
charge = [0,0] # for molecules A and B
unpaired_elec = [0,0]

BElib.orca_cal(cal_dir=cal_dir,input_xyz=input_xyz,MolA_name=MolA_name,MolB_name=MolB_name,N_dimer=N_dimer,functional=functional,basis_set=basis_set,charge=charge,unpaired_elec=unpaired_elec,orca_extra_keywords=orca_extra_keywords,NCore=NCore,orca_path=orca_path,openmpi_lib_path=openmpi_lib_path,openmpi_path=openmpi_path)
Eb = BElib.get_bind_ener(cal_dir=cal_dir,MolA_name=MolA_name,MolB_name=MolB_name,N_dimer=N_dimer)
print(Eb)
Eb.to_csv("binding_energy.csv")
exit()
xyz_coord = pd.read_csv('1.xyz',header=None,skiprows=2,delim_whitespace=True)

# Orca calculations
command = orca_path + " " + "test/water.inp"
err = lib.run_orca(command, "test/water.out")
#print(out)
print(err)
exit()
# xyz_coord for bsse a1: MolA; a2: molA + gh Orb; b3: MolB; b4: MolB + gh Orb
a1, a2, b3, b4 = lib.gen_bsse_xyz_coord(xyz_coord, 71)

# INPUT file for orca
lib.gen_orca_inp('hari.inp', xyz_coord, opt="OPT")#, "B3LYP/G", "6-31G(d,p)", "TightSCF grid5 NoFinalGrid ANGS", 0, 1)

# Calculate Binding Energy
PATH = '/home/hari/project/usa/huan/AB'
molA = 'a'
molB = 'b'
dimer = 5
Eb = lib.get_bind_ener(PATH,molA,molB,dimer)
print(Eb)
