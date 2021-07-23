import os
import dgl
import pickle
import sqlite3
import torch

from constants import *
from os.path import join
from pymetamap import MetaMap
from sqlitedict import SqliteDict
from utils import create_dir_if_not_exist

TEXT2GRAPH = pickle.loads(open(UMLS_TEXT2GRAPH_FILE, 'rb').read().replace(b'\r\n', b'\n'))

# Main Functions
def umls_search_concepts(sents, filtered_types = MM_TYPES):
    create_dir_if_not_exist(CACHE_DIR)
    search_results, cache_used, api_called = [], 0, 0
    sqlitedict = SqliteDict(UMLS_CONCEPTS_SQLITE, autocommit=True)
    for sent_idx, sent in enumerate(sents):
        if sent in sqlitedict:
            # Use cache
            cache_used += 1
            raw_concepts = sqlitedict[sent]
        else:
            # Use MetaMAP API
            api_called += 1
            METAMAP = MetaMap.get_instance(METAMAP_PATH)
            raw_concepts, error = METAMAP.extract_concepts([sent], [0])
            if error is None:
                sqlitedict[sent] = raw_concepts
            else:
                raise
        # Processing raw concepts
        processed_concepts = []
        for concept in raw_concepts:
            should_skip = False
            # Semantic Types
            if 'ConceptAA' in str(type(concept)):
                #print('Skipped ConceptAA')
                continue
            semtypes = set(concept.semtypes[1:-1].split(','))
            if len(semtypes.intersection(filtered_types)) == 0: continue # Skip
            semtypes = list(semtypes); semtypes.sort()
            # Offset Locations
            raw_pos_info = concept.pos_info
            raw_pos_info = raw_pos_info.replace(';',',')
            raw_pos_info = raw_pos_info.replace('[','')
            raw_pos_info = raw_pos_info.replace(']','')
            pos_infos = raw_pos_info.split(',')
            for pos_info in pos_infos:
                start, length = [int(a) for a in pos_info.split('/')]
                start_char = start - 1
                end_char = start+length-1
                # Heuristics Rules
                concept_text = sent[start_char:end_char]
                if concept_text == 'A': continue
                if concept_text == 'to': continue
                # Update processed_concepts
                processed_concepts.append({
                    'cui': concept.cui, 'semtypes': semtypes,
                    'start_char': start_char, 'end_char': end_char
                })
        search_results.append({
            'sent_idx': sent_idx, 'concepts': processed_concepts
        })
    sqlitedict.close()
    return search_results, {'cache_used': cache_used, 'api_called': api_called}

def umls_extract_network(sent):
    g_info = TEXT2GRAPH[sent]
    nodes, edges = list(set(g_info['nodes'])), list(set(g_info['edges']))

    # Build DGL graph
    graph_data = {}

    # Process edges
    edgetype2tensor1, edgetype2tensor2, edge_types = {}, {}, set()
    for n1, edge_type, n2 in edges:
        node1_index = nodes.index(n1)
        node2_index = nodes.index(n2)
        if not edge_type in edgetype2tensor1: edgetype2tensor1[edge_type] = []
        if not edge_type in edgetype2tensor2: edgetype2tensor2[edge_type] = []
        edgetype2tensor1[edge_type].append(node1_index)
        edgetype2tensor2[edge_type].append(node2_index)
        edge_types.add(edge_type)
    for edge_type in edge_types:
        graph_data[(NODE, edge_type, NODE)] = (torch.tensor(edgetype2tensor1[edge_type]),
                                               torch.tensor(edgetype2tensor2[edge_type]))

    # Finalize the graph
    G = dgl.heterograph(graph_data)
    assert(G.number_of_nodes() == len(nodes))

    return G, nodes
