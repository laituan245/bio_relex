# -*- coding: utf-8 -*-

from __future__ import print_function

import io
import os
import json
import argparse
import torch

from constants import *
from transformers import *
from models import JointModel
from utils import prepare_configs
from data import DataInstance, tokenize

def load_components(model_path, config_name = 'basic'):
    configs = prepare_configs(config_name, BIORELEX, 0)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])
    model = JointModel(configs)
    checkpoint = torch.load(model_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return tokenizer, model

def predict(model, tokenizer, sample):
    id, text = sample['id'], sample['text']
    test_sample = DataInstance(sample, id, text, tokenize(tokenizer, text.split(' ')))
    with torch.no_grad():
        pred_sample = model.predict(test_sample)
    return pred_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str,
                        help='Path to the pretrained model.')
    parser.add_argument('input_dir', type=str,
                        help='Path to directory containing input.json.')
    parser.add_argument('output_dir', type=str,
                        help='Path to output directory to write predictions.json in.')
    parser.add_argument('shared_dir', type=str,
                        help='Path to shared directory.')
    args = parser.parse_args()

    # Collect information on known relations
    self_path = os.path.realpath(__file__)
    self_dir = os.path.dirname(self_path)

    # Load main components
    tokenizer, model = load_components(args.model_dir)

    # Read input samples and predict w.r.t. set of relations.
    input_json_path = os.path.join(args.input_dir, 'input.json')
    output_json_path = os.path.join(args.output_dir, 'predictions.json')

    with io.open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = []
    for sample in data:
        sample = predict(model, tokenizer, sample)
        predictions.append(sample)

    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=True)


if __name__ == "__main__":
    main()
