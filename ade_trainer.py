import os
import copy
import utils
import torch
import json
import random
import math
import pyhocon
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from constants import *
from transformers import *
from data import load_data
from scorer import evaluate
from models import JointModel
from argparse import ArgumentParser
from trainer import train

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-s', '--start_nb', default=0)
    parser.add_argument('-e', '--end_nb', default=10)
    parser.add_argument('-c', '--config_name', default='basic')
    args = parser.parse_args()

    # Determine the range
    start_nb = int(args.start_nb); assert(0 <= start_nb and start_nb < 10)
    end_nb = int(args.end_nb); assert(0 <= end_nb and end_nb <= 10)
    split_nb_ranges = list(range(start_nb, end_nb))

    dev_scores = []
    for split_nb in split_nb_ranges:
        configs = prepare_configs(args.config_name, ADE, split_nb)
        configs['gradient_checkpointing'] = False
        dev_scores.append(train(configs))
    print('end of ade training')
    print(dev_scores)
    with open('ade_10_fold_results.json', 'w+') as f:
        f.write(json.dumps(dev_scores))
