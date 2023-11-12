import copy

from parser import Arguments
from main_exps import fl_training_ppm
from config import PASTEL
import argparse
import json
import time


if __name__ == '__main__':

    base_config = Arguments()

    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule_file')
    args = parser.parse_args()

    # Parse experiment schedule JSON file
    f = open(args.schedule_file)
    experiences_cfg = json.load(f)

    device = 'cuda' if base_config.gpu else 'cpu'

    for exp_cfg in experiences_cfg:

        try:

            print("Running experiment {}".format(exp_cfg['experiment_name']))
            start = time.time()
            # Update experiment configuration
            exp_running_config = copy.deepcopy(base_config)
            for key, value in exp_cfg['config'].items():
                setattr(exp_running_config, key, value)


            # Run experiment
            # exp_running_config.ppm = PASTEL
            fl_training_ppm(exp_running_config, device)
            end = time.time()

            print("Experiment {} successfully executed in {} .".format(str(exp_cfg['experiment_name']), str(end-start)))
            #
        except Exception as e:
            print("Experiment {} failed".format(exp_cfg['experiment_name']))
            print(e)


