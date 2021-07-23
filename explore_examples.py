import os
import json
import copy
import utils
import torch
import random
import math
import pyhocon
import warnings
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from constants import *
from transformers import *
from data import load_data
from argparse import ArgumentParser
from external_knowledge import umls_search_concepts

TEXT2GRAPH = pickle.loads(open(UMLS_TEXT2GRAPH_FILE, 'rb').read().replace(b'\r\n', b'\n'))
MIN_NB_NON_SINGLETONS = 1

def get_rcui2abbrv():
    rcui2abbrv = {}
    with open('resources/umls_reltypes.txt', 'r') as f:
        for line in f:
            abbrv, rcui = line.strip().split('|')[:2]
            rcui2abbrv[rcui] = abbrv
    return rcui2abbrv

def examine(split, entity_types, relation_types):
    rcui2abbrv = get_rcui2abbrv()
    for inst in split:
        # Original Infos
        text = inst.text
        print('\nTEXT: {}'.format(inst.text))
        print('\nENTITIES:')
        mentions, mention_locations = [], []
        for e in inst.entities:
            for m in e['mentions']:
                m_s, m_e = m['start_char'], m['end_char']
                m_type = entity_types[m['label']]
                print('{} | {} | start {} | end {}'.format(m_type, text[m_s:m_e], m_s, m_e))
                mentions.append(text[m_s:m_e])
                mention_locations.append((m_s, m_e))
        print('\nRELATIONS:')
        for r in inst.relations:
            p1, p2 = r['participants']
            r_label = relation_types[r['label']]
            print('{} between [{}] and [{}]'.format(r_label, mentions[p1], mentions[p2]))
        # Retrieve UMLS graph
        graph = TEXT2GRAPH[text]
        nodes, edges = graph['nodes'], graph['edges']
        # Map node to texts and semtypes
        print('\nNodes:')
        kg_concepts = umls_search_concepts([text])[0][0]['concepts']
        for n in nodes:
            node_texts, node_semtypes = [], []
            for c in kg_concepts:
                if n == c['cui']:
                    node_texts.append(text[c['start_char']:c['end_char']])
                    node_semtypes += c['semtypes']
            node_texts, node_semtypes = set(node_texts), set(node_semtypes)
            if len(node_texts) == 0 and len(node_semtypes) == 0: continue
            print('node {}: texts = {} | semtypes = {}'.format(n, node_texts, node_semtypes))
        # Map location to concepts
        print('\nMentions Mapped to Multiple Concepts')
        loc2concepts = {}
        for c in kg_concepts:
            loc = c['start_char'], c['end_char']
            if not loc in mention_locations: continue
            if not loc in loc2concepts: loc2concepts[loc] = []
            loc2concepts[loc].append(c)
        nb_non_singletons = 0
        for loc in loc2concepts:
            m_text = text[loc[0]:loc[1]]
            if len(loc2concepts[loc]) > 1:
                nb_non_singletons += 1
                ambiguous_tag = '[AMBIGUOUS]'
            else:
                ambiguous_tag = '[NON AMBIGUOUS]'
            print('{} | Mention {} | Concepts {}'.format(ambiguous_tag, m_text, loc2concepts[loc]))
        if nb_non_singletons < MIN_NB_NON_SINGLETONS: continue
        if not len(loc2concepts) == len(mention_locations): continue

        # nx.MultiDiGraph to visualize
        DG = nx.MultiDiGraph()
        for n in nodes: DG.add_node(n)
        for n1, rcui, n2 in edges:
            DG.add_edge(n1, n2, rcui= rcui, abbrv= rcui2abbrv[rcui])
        nx.draw(DG, with_labels=True, font_weight='bold')
        plt.show()
        input('\nContinue?\n')

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='basic')
    parser.add_argument('-d', '--dataset', default=ADE, choices=DATASETS)
    parser.add_argument('-s', '--split_nb', default=0) # Only affect ADE dataset
    args = parser.parse_args()
    args.split_nb = int(args.split_nb)

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
    train, dev = load_data(args.dataset, args.split_nb, tokenizer)
    print('Loaded {} dataset (Train {} | Dev {})'.format(args.dataset, len(train), len(dev)))
    if args.dataset == ADE:
        entity_types = ADE_ENTITY_TYPES
        relation_types = ADE_RELATION_TYPES
    elif args.dataset == BIORELEX:
        entity_types = BIORELEX_ENTITY_TYPES
        relation_types = BIORELEX_RELATION_TYPES
    print('\n')

    # Examine
    examine(train, entity_types, relation_types)
