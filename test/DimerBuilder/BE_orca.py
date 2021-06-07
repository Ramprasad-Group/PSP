import numpy as np
import pandas as pd
import psp.BE_lib as BElib
import mmap
import io
import re
import os

# inputs
HOME_DIR = os.getcwd()
input_xyz = os.path.join(HOME_DIR,'input-xyz')
MolA_name = 'A'
MolB_name = 'B'
cal_dir = os.path.join(HOME_DIR,'dft_calc_orca')
N_dimer = 4
NCore = 8
functional = "B3LYP/G"
basis_set = "6-31G(d,p)"
orca_extra_keywords = "D3BJ TightSCF grid5 NoFinalGrid ANGS"
charge = [0,0] # for molecules A and B
unpaired_elec = [0,0]

BElib.orca_cal(cal_dir=cal_dir,input_xyz=input_xyz,MolA_name=MolA_name,MolB_name=MolB_name,N_dimer=N_dimer,functional=functional,basis_set=basis_set,charge=charge,unpaired_elec=unpaired_elec,orca_extra_keywords=orca_extra_keywords,NCore=NCore)
Eb = BElib.get_bind_ener_orca(cal_dir=cal_dir,MolA_name=MolA_name,MolB_name=MolB_name,N_dimer=N_dimer)
print(Eb)
Eb.to_csv("binding_energy_orca.csv")
