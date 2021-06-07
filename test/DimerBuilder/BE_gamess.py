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
cal_dir = os.path.join(HOME_DIR,'dft_calc_gamess')
N_dimer = 4
gamess_path = "/home/hari/.soft/gamess/rungms"
scrach_dir = "/home/hari/test/gamessSRC"

# Gamess input
basis = " $BASIS GBASIS=N31 NGAUSS=6 NDFUNC=1 NPFUNC=1 $END"
func_opt = " $CONTRL SCFTYP=RHF RUNTYP=OPTIMIZE DFTTYP=B3LYP MULT=1 $END"
func_sp = " $CONTRL SCFTYP=RHF RUNTYP=ENERGY DFTTYP=B3LYP MULT=1 $END"
disp = " $DFT DC=.TRUE. IDCVER=4 $END"
opt_par_constA = " $STATPT OPTTOL=0.0001 NSTEP=500 IFREEZ(1)=34,35,36,196,197,198 $END"
opt_par = " $STATPT OPTTOL=0.0001 NSTEP=500 $END"
mem = " $SYSTEM MWORDS=625 $END"

keywords_MolA_opt = [basis,func_opt,opt_par,mem]
keywords_MolA_sp = [basis,func_sp,mem]

keywords_MolB_opt = [basis,func_opt,opt_par,mem]
keywords_MolB_sp = [basis,func_sp,mem]

keywords_MolAB_opt = [basis,func_opt,opt_par,mem]
keywords_MolAB_sp = [basis,func_sp,mem]

keywords_dict = {"MolA_opt": keywords_MolA_opt, "MolA_sp": keywords_MolA_sp, "MolB_opt": keywords_MolB_opt, "MolB_sp": keywords_MolB_sp, "MolAB_opt": keywords_MolAB_opt, "MolAB_sp": keywords_MolAB_sp}

BElib.gamess_cal(cal_dir=cal_dir,input_xyz=input_xyz,MolA_name=MolA_name,MolB_name=MolB_name,N_dimer=N_dimer, gamess_path=gamess_path, scrach_dir=scrach_dir, keywords_dict=keywords_dict)
Eb = BElib.get_bind_ener_gamess(cal_dir=cal_dir, MolA_name=MolA_name, MolB_name=MolB_name, N_dimer=N_dimer)
print(Eb)
Eb.to_csv("binding_energy_gamess.csv")
