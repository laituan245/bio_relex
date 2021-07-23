import os
import copy
import utils
import json
import torch
import random
import math
import pyhocon
import warnings
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import *
from constants import *
from transformers import *
from data import load_data
from scorer import evaluate
from models import JointModel
from argparse import ArgumentParser

SHOW_GRAPHS = True

# Helper Functions
def is_same_graph(g1, g2):
    g1_nodes, g1_edges = g1['nodes'], g1['edges']
    g2_nodes, g2_edges = g2['nodes'], g2['edges']

    # Check nodes
    for n in g1_nodes:
        if not n in g2_nodes:
            return False
    for n in g2_nodes:
        if not n in g1_nodes:
            return False
    # Check edges
    for e in g1_edges:
        if not e in g2_edges:
            return False
    for e in g2_edges:
        if not e in g1_edges:
            return False

    return True

def build_and_save_graph(g, title):

    # Build networkx DiGraph
    G = nx.DiGraph()
    # Add nodes
    node_colors = []
    for n in g['nodes']:
        n_name = '{}_{}'.format(n['start'], n['end'])
        G.add_node(n_name)
        color = 'red' if n['label'] == 'Drug' else 'yellow' # ADE-specific
        node_colors.append(color)

    # Add edges
    di_edges = []
    for e in g['edges']:
        di_edges.append((e['head'], e['tail']))
    G.add_edges_from(di_edges)

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G, scale=0.01)
    nx.draw_networkx_nodes(G, pos, node_color = node_colors)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)
    plt.title(title)
    plt.savefig('{}.jpg'.format(title))
    plt.close()

# Main Code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--truth_graphs', default='sent2truthgraph.json')
    parser.add_argument('--first_phase_graphs', default='sent2predgraph_first.json')
    parser.add_argument('--full_model_graphs', default='sent2predgraph_full.json')
    args = parser.parse_args()

    # Load graphs
    with open(args.truth_graphs, 'r') as f: sent2truthgraph = json.load(f)
    with open(args.first_phase_graphs, 'r') as f: sent2graph_first = json.load(f)
    with open(args.full_model_graphs, 'r') as f: sent2graph_full = json.load(f)
    all_texts = list(sent2truthgraph.keys())

    # Examine text
    log_file = open('graph_diff_logs.txt', 'w+', encoding='utf-8')
    for text in all_texts:
        truth_graph = sent2truthgraph[text]
        pred_graph_first = sent2graph_first[text]
        pred_graph_full = sent2graph_full[text]

        if is_same_graph(pred_graph_full, pred_graph_first): continue
        if not is_same_graph(truth_graph, pred_graph_full): continue

        # Write to log_file
        log_file.write('{}\n'.format(text))
        log_file.write('Ground-Truth Graph\n')
        log_file.write('{}\n'.format(json.dumps(truth_graph)))
        log_file.write('1st-phase Graph\n')
        log_file.write('{}\n'.format(json.dumps(pred_graph_first)))
        log_file.write('2nd-phase Graph\n')
        log_file.write('{}\n'.format(json.dumps(pred_graph_full)))
        log_file.write('\n')
        log_file.flush()

        # Show Graphs
        if SHOW_GRAPHS:
            print(text)
            print('1st-phase graph')
            build_and_save_graph(pred_graph_first, '1st-phase graph')
            print('2nd-phase graph')
            build_and_save_graph(pred_graph_full, '2nd-phase graph')
            input('Continue?')

    log_file.close()
