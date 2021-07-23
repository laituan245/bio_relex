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
from os.path import join
from spacy import displacy
from data import load_data
from scorer import evaluate
from models import JointModel
from argparse import ArgumentParser
from collections import Counter
from scorer.ade import get_relation_mentions
from external_knowledge import umls_search_concepts

SHOW_ERRORS_ONLY = True

def get_entity_mentions(sentence):
    typed_mentions = []
    for cluster in sentence['entities']:
        for alias, entity in cluster['names'].items():
            for start, end in entity['mentions']:
                if entity['is_mentioned']:
                    typed_mentions.append({'start': start, 'end': end, 'label': cluster['label']})
    typed_mentions.sort(key=lambda x: x['start'])
    return typed_mentions

def graph_from_sent(sent_text, sent):
    nodes, edges = [], []

    ents = get_entity_mentions(sent)
    for e in ents:
        e['text'] = sent_text[e['start']:e['end']]
        nodes.append(e)

    rels = get_relation_mentions(sent)
    for r in rels:
        head_loc = [int(l) for l in r['head'].split('_')]
        tail_loc = [int(l) for l in r['tail'].split('_')]
        r['head_text'] = sent_text[head_loc[0]:head_loc[1]]
        r['tail_text'] = sent_text[tail_loc[0]:tail_loc[1]]
        edges.append(r)

    return {'nodes': nodes, 'edges': edges}

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-m', '--trained_model', default='model.pt')
    parser.add_argument('-c', '--config_name', default='basic')
    parser.add_argument('-d', '--dataset', default=BIORELEX, choices=DATASETS)
    parser.add_argument('-s', '--split_nb', default=0) # Only affect ADE dataset
    args = parser.parse_args()
    args.split_nb = int(args.split_nb)

    # Reload components
    configs = prepare_configs(args.config_name, args.dataset, args.split_nb)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])
    train, dev = load_data(configs['dataset'], configs['split_nb'], tokenizer)
    model = JointModel(configs)

    # Reload a model
    assert (os.path.exists(args.trained_model))
    checkpoint = torch.load(args.trained_model, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print('Reloaded a pretrained model')

    # Evaluation on the dev set
    print('Evaluation on the dev set')
    if configs['use_external_knowledge']:
        model.knowledge_enhancer.start_logging()
    evaluate(model, dev, configs['dataset'])
    if configs['use_external_knowledge']:
        model.knowledge_enhancer.end_logging()

    # Visualize predictions and groundtruths (on the dev set)
    truths, preds, sent2truthgraph, sent2predgraph = [], [], {}, {}
    total_ents, ents_covered_by_metamap = 0, 0
    total_fns, fn_covered_by_metamap = 0, 0
    relevants_covered_by_metamap, non_relevants_covered_by_metamap = [], []
    cuid2embs = pickle.load(open(UMLS_EMBS, 'rb'))
    with torch.no_grad():
        for i in range(len(dev)):
            truth_sentence = dev[i].data
            truth_ents = get_entity_mentions(truth_sentence)
            pred_sentence = model.predict(dev[i])
            pred_ents = get_entity_mentions(pred_sentence)

            # Update sent2truthgraph and sent2predgraph
            sent_text = dev[i].text
            sent2truthgraph[sent_text] = graph_from_sent(sent_text, truth_sentence)
            sent2predgraph[sent_text] = graph_from_sent(sent_text, pred_sentence)

            # check if the prediction is the same as the annotation
            if SHOW_ERRORS_ONLY:
                typed_truths = set({(x['start'], x['end'], x['label']) for x in truth_ents})
                typed_preds = set({(x['start'], x['end'], x['label']) for x in pred_ents})
                if typed_truths == typed_preds:
                    # Skip
                    continue

            # Update truths
            truths.append({
                'text': truth_sentence['text'], 'title': None,
                'ents': truth_ents,
            })
            # Update preds
            preds.append({
                'text': pred_sentence['text'], 'title': None,
                'ents': pred_ents,
            })

            # Investigate the usefulness of MetaMap
            text = truth_sentence['text']
            umls_concepts = umls_search_concepts([text])[0][0]['concepts']
            umls_concepts = [c for c in umls_concepts if c['cui'] in cuid2embs]
            loc_umls = set({(x['start_char'], x['end_char']) for x in umls_concepts})
            loc_truths = set({(x['start'], x['end']) for x in truth_ents})
            loc_preds = set({(x['start'], x['end']) for x in pred_ents})
            false_negatives = loc_truths - loc_preds
            # Compute number of ground-truth entities covered by MetaMap
            total_ents += len(loc_truths)
            ents_covered_by_metamap += len(loc_truths.intersection(loc_umls))
            for c in umls_concepts:
                if (c['start_char'], c['end_char']) in loc_truths:
                    relevants_covered_by_metamap.append(c)
            # Compute number of false negatives covered by MetaMap
            total_fns += len(false_negatives)
            fn_covered_by_metamap += len(false_negatives.intersection(loc_umls))
            # Compute number of MetaMap concepts that are not considered as entities in the dataset
            for c in umls_concepts:
                if not (c['start_char'], c['end_char']) in loc_truths:
                    non_relevants_covered_by_metamap.append(c)

    # Output sent2truthgraph, sent2predgraph
    with open('sent2truthgraph.json', 'w+') as f:
        f.write(json.dumps(sent2truthgraph))
    with open('sent2predgraph.json', 'w+') as f:
        f.write(json.dumps(sent2predgraph))

    # Write out relevant types
    relevant_types = flatten([c['semtypes'] for c in relevants_covered_by_metamap])
    with open('relevant_types.txt', 'w+') as f:
        f.write(json.dumps(Counter(relevant_types)))
    # Write out non relevant types
    non_relevant_types = flatten([c['semtypes'] for c in non_relevants_covered_by_metamap])
    with open('non_relevant_types.txt', 'w+') as f:
        f.write(json.dumps(Counter(non_relevant_types)))
    print('types can be discarded: {}'.format(set(non_relevant_types) - set(relevant_types)))
    print('non_relevants_covered_by_metamap = {}'.format(len(non_relevants_covered_by_metamap)))
    print('ents_covered_by_metamap = {} | total_ents = {}'.format(ents_covered_by_metamap, total_ents))
    print('fn_covered_by_metamap = {} | total_fns = {}'.format(fn_covered_by_metamap, total_fns))

    # Generate html file
    output_dir = 'visualizations/{}_{}'.format(args.dataset, args.split_nb)
    os.makedirs(output_dir, exist_ok=True)
    truth_html = displacy.render(truths, style="ent", page=True, manual=True)
    pred_html = displacy.render(preds, style="ent", page=True, manual=True)
    with open(join(output_dir, 'truths.html'), 'w+', encoding='utf-8') as f:
        f.write(truth_html)
    with open(join(output_dir, 'preds.html'), 'w+', encoding='utf-8') as f:
        f.write(pred_html)
