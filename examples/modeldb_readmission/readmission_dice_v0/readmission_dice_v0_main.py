import os
import argparse

# try:
#     import verta
# except ImportError:
#     !pip install verta

import verta
from verta import Client
from verta.utils import ModelAPI


HOST = "http://localhost:3009"
PROJECT_NAME = "readmission_dice_v0"
EXPERIMENT_NAME = "readmission_dice_v0_first_run"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Template Main Function')
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--output', type=str, help='output folder')
    parser.add_argument('--vis', type=str, help='vis folder')
    args = parser.parse_args()

    client = Client(HOST)
    proj = client.set_project(PROJECT_NAME)
    expt = client.set_experiment(EXPERIMENT_NAME)
    run = client.set_experiment_run()

    cmd = "readmission_dice_trainvalidationtest.sh"  # modify
    lib_param = {}
    lib_param["--input"] = args.input
    lib_param["--output"] = args.output
    lib_param["--vis"] = args.vis

    for k, v in lib_param.items():
        cmd = cmd + " " + str(k) + " " + str(v)
    
    cmd = "bash " + cmd
    print ("executing cmd: \n", cmd)
    os.system(cmd)  # modify

    # run.log_setup_script(cmd[:10])
