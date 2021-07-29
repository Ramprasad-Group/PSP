import numpy as np
import pandas as pd
import psp.PSP_lib as bd
import psp.BE_lib as BElib
from scipy.spatial.distance import cdist
import psp.MoleculeBuilder as mb
import os
import shutil


class Builder:
    def __init__(
        self,
        Dataframe,
        ID_col='ID',
        SMILES_col='smiles',
        Length=[1],
        NumConf=1,
        Ndimer=10,
        ABdis=2.0,
        Loop=False,
        OutFile='',
        OutDir='dimer_models',
        OutDir_xyz='molecules',
    ):
        self.Dataframe = Dataframe
        self.ID_col = ID_col
        self.SMILES_col = SMILES_col
        self.Length = Length
        self.NumConf = NumConf
        self.Ndimer = Ndimer
        self.ABdis = ABdis
        self.Loop = Loop
        self.OutFile = OutFile
        self.OutDir = OutDir
        self.OutDir_xyz = OutDir_xyz

    def Build(self):
        # location of directory for VASP inputs (polymers) and build a directory
        out_dir = os.path.join(self.OutDir, "")
        bd.build_dir(out_dir)
        OutDir_xyz = os.path.join(out_dir, self.OutDir_xyz, "")
        bd.build_dir(OutDir_xyz)

        xyz_gen_pd = pd.DataFrame()
        for i in self.Dataframe.index:
            df = pd.DataFrame(self.Dataframe.loc[i]).T
            mol = mb.Builder(
                df,
                ID_col=self.ID_col,
                SMILES_col=self.SMILES_col,
                OutDir=OutDir_xyz,
                Length=self.Length,
                NumConf=1,
                Loop=self.Loop,
            )
            results = mol.Build()
            xyz_gen_pd = pd.concat([xyz_gen_pd, results])

        if len(list(set(xyz_gen_pd['Result'].values))) != 1:
            xyz_gen_pd.to_csv("molecules.csv")
            print(
                "Couldn't generate XYZ coordinates of molecules, check 'molecules.csv'"
            )
            exit()

        XYZ_list = []
        for index, row in self.Dataframe.iterrows():
            XYZ_list.append(
                OutDir_xyz
                + str(row[self.ID_col])
                + '_N'
                + str(self.Length[0])
                + '_C1.xyz'
            )

        # PATH and name of XYZ files of molecules A and B
        NnameUnitA = XYZ_list[0]
        NnameUnitB = XYZ_list[1]

        # Read XYZ coordinates
        unitA = pd.read_csv(NnameUnitA, header=None, skiprows=2, delim_whitespace=True)
        unitB = pd.read_csv(NnameUnitB, header=None, skiprows=2, delim_whitespace=True)

        # Gen XYZ coordinates for molecules A and B
        bd.gen_xyz(out_dir + 'A' + ".xyz", unitA)
        bd.gen_xyz(out_dir + 'B' + ".xyz", unitB)

        # Get minimum and maximum in X, Y and Z axes for molecule A
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = (
            unitA[1].min(),
            unitA[1].max(),
            unitA[2].min(),
            unitA[2].max(),
            unitA[3].min(),
            unitA[3].max(),
        )
        Xdis, Ydis, Zdis = Xmax - Xmin, Ymax - Ymin, Zmax - Zmin

        # Create a Dataframe for molecule A size
        SmolA = pd.DataFrame()
        SmolA['Min'] = [Xmin, Ymin, Zmin]
        SmolA['Max'] = [Xmax, Ymax, Zmax]
        SmolA['Dis'] = [Xdis, Ydis, Zdis]

        # rename index to match with columns of XYZ coordinates (unitA)
        SmolA.index = [1, 2, 3]

        # Keep a copy the Dataframe for later use
        SmolA_copy = SmolA.copy()

        # sort and select the longest axis
        SmolA = SmolA.sort_values(by='Dis', ascending=False).head(1)

        # list other two axes
        other_axes = [1, 2, 3]
        other_axes.remove(SmolA.index.values[0])

        # Check distance between A and B and increase it if necessary
        def adjust_unitB(unitA, unitB_mod, ABdis, down=False):
            dist = cdist(unitA[[1, 2, 3]].values, unitB_mod[[1, 2, 3]].values)
            while (dist < ABdis).any():
                adj_dis = ABdis + 0.1 - np.min(dist)
                if down is False:
                    unitB_mod = BElib.move_barycenter(
                        unitB_mod,
                        [adj_dis, adj_dis, adj_dis],
                        origin=False,
                        barycenter=False,
                    )
                else:
                    unitB_mod = BElib.move_barycenter(
                        unitB_mod,
                        [-adj_dis, -adj_dis, -adj_dis],
                        origin=False,
                        barycenter=False,
                    )
                dist = cdist(unitA[[1, 2, 3]].values, unitB_mod[[1, 2, 3]].values)
            return unitB_mod

        # Divide molecule A into Ndimer/2 parts
        part_dis = SmolA.Dis.values[0] / (int(self.Ndimer / 2))
        part_min = SmolA.Min.values[0]
        count = 1
        for i in range(int(self.Ndimer / 2)):
            part_max = part_min + part_dis
            part_xyz = unitA.loc[
                (unitA[SmolA.index.values[0]] > part_min)
                & (unitA[SmolA.index.values[0]] < part_max)
            ]
            # Calculate mid of max and min for the part of the molecule A
            part_mid = part_min + (part_max - part_min) / 2
            # Get subparts; in up, we will consider max of other two axes and vice versa
            sub_part_up = part_xyz.loc[
                (part_xyz[SmolA.index.values[0]] > part_min)
                & (part_xyz[SmolA.index.values[0]] < part_mid)
            ]
            sub_part_down = part_xyz.loc[
                (part_xyz[SmolA.index.values[0]] > part_mid)
                & (part_xyz[SmolA.index.values[0]] < part_max)
            ]

            # Move molecule B near to origin; note that all coordinates are positive
            unitB = BElib.move_barycenter(unitB, '', origin=True, barycenter=False)

            # Get an atom with Max of two other axes
            # UP
            dis_axis1_up = (
                SmolA_copy.loc[other_axes[0]].Max - sub_part_up[other_axes[0]].max()
            )
            dis_axis2_up = (
                SmolA_copy.loc[other_axes[1]].Max - sub_part_up[other_axes[1]].max()
            )
            if dis_axis1_up <= dis_axis2_up:
                atom_up = sub_part_up[
                    sub_part_up[other_axes[0]] == sub_part_up[other_axes[0]].max()
                ]
                unitB[other_axes[0]] = unitB[other_axes[0]] + self.ABdis

            else:
                atom_up = sub_part_up[
                    sub_part_up[other_axes[1]] == sub_part_up[other_axes[1]].max()
                ]
                unitB[other_axes[1]] = unitB[other_axes[1]] + self.ABdis

            unitB_up = BElib.move_barycenter(
                unitB,
                [atom_up[1].values[0], atom_up[2].values[0], atom_up[3].values[0]],
                origin=False,
                barycenter=False,
            )
            # Check distance between A and B and increase it if necessary
            unitB_up = adjust_unitB(unitA, unitB_up, self.ABdis, down=False)

            # Combine with unitA
            unitAB_up = pd.concat([unitA, unitB_up])
            bd.gen_xyz(out_dir + self.OutFile + str(count) + ".xyz", unitAB_up)

            # Move molecule B near to origin; note that all coordinates are positive
            unitB = BElib.move_barycenter(unitB, '', origin=True, barycenter=False)

            # DOWN
            dis_axis1_down = (
                sub_part_down[other_axes[0]].min() - SmolA_copy.loc[other_axes[0]].Min
            )
            dis_axis2_down = (
                sub_part_down[other_axes[1]].min() - SmolA_copy.loc[other_axes[1]].Min
            )
            if dis_axis1_down <= dis_axis2_down:
                atom_down = sub_part_down[
                    sub_part_down[other_axes[0]] == sub_part_down[other_axes[0]].min()
                ]
                unitB[other_axes[0]] = unitB[other_axes[0]] - self.ABdis

            else:
                atom_down = sub_part_down[
                    sub_part_down[other_axes[1]] == sub_part_down[other_axes[1]].min()
                ]
                unitB[other_axes[1]] = unitB[other_axes[1]] - self.ABdis

            unitB_down = BElib.move_barycenter(
                unitB,
                [
                    atom_down[1].values[0],
                    atom_down[2].values[0],
                    atom_down[3].values[0],
                ],
                origin=False,
                barycenter=False,
            )

            # Check distance between A and B and increase it if necessary
            unitB_down = adjust_unitB(unitA, unitB_down, self.ABdis, down=True)

            # Combine with unitA
            unitAB_down = pd.concat([unitA, unitB_down])
            bd.gen_xyz(out_dir + self.OutFile + str(count + 1) + ".xyz", unitAB_down)

            part_min = part_max
            count += 2

        # Delete INPUT xyz directory
        if os.path.isdir(OutDir_xyz):
            shutil.rmtree(OutDir_xyz)
