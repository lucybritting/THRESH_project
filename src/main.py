import argparse

from utils.pipeline import *

# TODO: note somewhere whether it was done with top100features or all features?

parser = argparse.ArgumentParser()
parser.add_argument("--cohorts", nargs='+', required=True)
parser.add_argument("--range_merge", nargs='+', default='widest', choices=['widest', 'average', 'narrowest'])
parser.add_argument("--top100labs", action='store_true')  # set if want to use only the top 100 labs, not all labs
args = parser.parse_args()

# scan labevents for continuous and binarised values and ranges
run_scan_step(args)
# merge multi ranges
run_merge_step(args)
# create discretised mapping
run_mapping_step(args)


# TODO: separate the cohorts into 5 folds and train-test sets

# TODO: implement imputation
# TODO: implement oversampling

# TODO: train the selected classifiers on all mappings (including concat) and save the models as well as the average performance measures
