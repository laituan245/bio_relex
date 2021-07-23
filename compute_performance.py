import os
import spacy
import copy
import utils
import torch
import random
import math
import json
import pickle
import pyhocon
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from constants import *
from transformers import *
from os import listdir
from os.path import isfile, join
from spacy import displacy
from data import load_data
from scorer import evaluate
from models import JointModel
from argparse import ArgumentParser
from collections import Counter
from external_knowledge import umls_search_concepts

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-m', '--models_dir', required=True)
    args = parser.parse_args()

    # Extract models_dir, dataset, and config_name
    models_dir = args.models_dir
    dir_name = os.path.basename(os.path.normpath(models_dir))
    parts = dir_name.split('_')
    dataset, config_name = parts[0], '_'.join(parts[1:])
    print('models_dir = {} | dataset = {} | config_name = {}'.format(models_dir, dataset, config_name))

    # Evaluate each model in the directory
    mention_scores, relation_scores = [], []
    for f in listdir(models_dir):
        if isfile(join(models_dir,f)) and f.endswith('.pt'):
            trained_model_path = join(models_dir,f)
            split_nb = int(f.split('_')[1].split('.')[0])
            print('Split {}'.format(split_nb))
            # Reload components 
            configs = prepare_configs(config_name, dataset, split_nb, verbose=False)
            tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])
            train, dev = load_data(configs['dataset'], configs['split_nb'], tokenizer)
            model = JointModel(configs)
            checkpoint = torch.load(trained_model_path, map_location=model.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # Evaluation on the dev set
            dev_m_score, dev_rel_score = evaluate(model, dev, configs['dataset'])
            mention_scores.append(dev_m_score)
            relation_scores.append(dev_rel_score)
            # Free resources
            del model
            torch.cuda.empty_cache()
    # Print Overall Results
    multiplier = 100 if dataset == BIORELEX else 1
    m_avg = round(multiplier * np.average(mention_scores), 2)
    r_avg = round(multiplier * np.average(relation_scores), 2)
    print('\nOverall')
    print('Entity Recognition: Avg = {} | Std = {}'.format(m_avg, np.std(mention_scores)))
    print('Relation Extraction: Avg = {} | Std = {}'.format(r_avg, np.std(relation_scores)))
