import os


if __name__ == '__main__':
    cmd = "cd readmission_dice_v0/ && python readmission_dice_v0_main.py --input /hdd2/feiyi/mlflow/multistep_readmission/dataset_nuh_read_q5/demographic.csv,/hdd2/feiyi/mlflow/multistep_readmission/dataset_nuh_read_q5/cases.csv --output ./output/ --vis ./vis/ && cd .."
    os.system(cmd)

    cmd = "cd readmission_extractor_v0/ && python readmission_extractor_v0_main.py --input ../readmission_dice_v0/output/ --output ./output/ --vis ./vis/ && cd .."
    os.system(cmd)

    cmd = "cd readmission_shared_data_preprocess_v0/ && python readmission_shared_data_preprocess_v0_main.py --input ../readmission_extractor_v0/output/ --output ./output/ --vis ./vis/ && cd .."
    os.system(cmd)

    cmd = "cd readmission_CNN_v0/ && python readmission_CNN_v0_main.py --input ../readmission_shared_data_preprocess_v0/output/ --output ./output/ --vis ./vis/ --trainratio 1.0 --validationratio 0.0 --testratio 0.0 && cd .."
    os.system(cmd)
