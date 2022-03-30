import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)


def print_psp_info():
    print("")
    print(
        "                        ---------  PPPPPP       SSSSSS    PPPPPP    ---------                        "
    )
    print(
        "                -----------------  PP    PP   SS          PP    PP  -----------------                "
    )
    print(
        "        -------------------------  PP    PP   SS          PP    PP  -------------------------        "
    )
    print(
        " --------------------------------  PPPPPP       SSSSS     PPPPPP    -------------------------------- "
    )
    print(
        "        -------------------------  PP                SS   PP        -------------------------        "
    )
    print(
        "                -----------------  PP                SS   PP        -----------------                "
    )
    print(
        "                        ---------  PP          SSSSSS     PP        ---------                        "
    )
    print(
        " --------------------------------------------------------------------------------------------------- "
    )
    version = '1.0.0'  # pkg_resources.require("PolymerStructurePredictor")[0].version
    print(
        "          ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **          "
    )
    print(
        "                        Polymer Structure Predictor (PSP) version = ",
        version,
        "                    ",
    )
    print(
        "                                   Developed at Ramprasad Group                                      "
    )
    print(
        "           Materials Science and Engineering, Georgia Institute of Technology, Atlanta, US           "
    )
    print("")
    print(
        "           Cite this work as:                                                                "
    )
    print(
        "           H. Sahu, K-H. Shen, J. H. Montoya, H. Tran, R. Ramprasad, PSP: A python toolkit           "
    )
    print(
        "           for predicting 3D models of polymers, journal name, volume, page, 2022.                   "
    )

    print(
        "          ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **          "
    )
    print(
        " --------------------------------------------------------------------------------------------------- "
    )


def print_input(builder, input_file=pd.DataFrame()):
    print(" ", builder, " started...")
    if not input_file.empty:
        print(
            " ----------------------------------------------- INPUT --------------------------------------------- "
        )
        input_file.index = np.arange(1, len(input_file) + 1)
        print(input_file.to_markdown())


def print_out(output_file, model_name, time, subscript=False):
    print("\n", model_name, "building completed.\n")
    if subscript is True or model_name == "Amorphous model":
        pass
    else:
        print(
            " ----------------------------------------------- OUTPUT -------------------------------------------- "
        )
    if not output_file.empty:
        output_file.index = np.arange(1, len(output_file) + 1)
        print(output_file.to_markdown())
        print("")
    if subscript is False:
        print(" Total run time (minutes): ", time)
        print(
            " ------------------------------------- PSP TERMINATED NORMALLY ------------------------------------- "
        )
    else:
        print("", model_name, "building time (minutes): ", time)
        print("")
