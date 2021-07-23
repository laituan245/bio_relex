import dgl
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from utils import *
from constants import *
from models.base import *
from models.helpers import *
from external_knowledge import *
from models.graph.rgcn import RGCNModel
from models.graph.gcn import GraphConvolution

class AttentionBasedLinkingModule(nn.Module):
    def __init__(self, configs):
        super(AttentionBasedLinkingModule, self).__init__()
        self.configs = configs

        d_model = configs['span_emb_size']
        dropout = configs['dropout_rate']

        # Sentinel Features
        self.sentinel_linear = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()

        # Attention Scorer
        self.attention_scorer = FFNNModule(input_size = 2 * d_model,
                                           hidden_sizes = [500, 500],
                                           output_size = 1,
                                           dropout = dropout)

    def forward(self, span_embs, knowledge_embs, span2concepts):
        nb_candidates = span_embs.size()[0]
        nb_concepts = knowledge_embs.size()[0]     # Number of knowledge concepts

        # Sentinel Features
        sentinel_embs = self.relu(self.sentinel_linear(span_embs))
        sentinel_embs = sentinel_embs.unsqueeze(1)

        # Compute unnormalized attention scores
        span_embs = span_embs.unsqueeze(1).repeat([1, nb_concepts+1, 1])
        knowledge_embs = knowledge_embs.unsqueeze(0).repeat([nb_candidates, 1, 1])
        knowledge_embs = torch.cat([knowledge_embs, sentinel_embs], dim=1)
        concat_embs = torch.cat([span_embs, knowledge_embs], dim=-1)
        unnormalized_scores = self.attention_scorer(concat_embs).view(nb_candidates, nb_concepts+1)

        # Compute features
        features, attention_probs = [], []
        for i in range(nb_candidates):
            concept_indexes = span2concepts[i] + [nb_concepts]
            concept_indexes = torch.LongTensor(concept_indexes).to(self.device)
            relevant_embs = torch.index_select(knowledge_embs[i,:,:], 0, concept_indexes)
            relevant_scores = torch.index_select(unnormalized_scores[i,:], 0, concept_indexes)
            relevant_probs = torch.softmax(relevant_scores, dim=-1).unsqueeze(1)
            # Update features and attention_probs
            features.append(torch.sum(relevant_probs * relevant_embs, dim=0))
            attention_probs.append(tolist(relevant_probs.squeeze(1)))
        features = torch.cat([f.unsqueeze(0) for f in features], dim=0)

        return features, attention_probs

class BiGCNLayer(nn.Module):
    def __init__(self, etypes, configs):
        super(BiGCNLayer, self).__init__()
        self.etypes = etypes
        self.num_rels = len(etypes)
        self.configs = configs
        self.hid_size = configs['span_emb_size']

        gcn2p_fw, gcn2p_bw = [], []
        for _ in range(self.num_rels):
            gcn2p_fw.append(GraphConvolution(self.hid_size, self.hid_size // 2))
            gcn2p_bw.append(GraphConvolution(self.hid_size, self.hid_size // 2))
        self.gcn2p_fw = nn.ModuleList(gcn2p_fw)
        self.gcn2p_bw = nn.ModuleList(gcn2p_bw)

        self.dropout = nn.Dropout(configs['ieg_bignn_dropout'])
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.hid_size, self.hid_size)

    def forward(self, inps, fw_adjs, bw_adjs):
        # No self-loops in fw_adjs or bw_adjs
        num_rels = self.num_rels
        assert(len(fw_adjs) == num_rels)
        assert(len(bw_adjs) == num_rels)

        outs = []
        for i in range(num_rels):
            fw_outs = self.gcn2p_fw[i](inps, fw_adjs[i])
            bw_outs = self.gcn2p_bw[i](inps, bw_adjs[i])
            outs.append(self.dropout(torch.cat([bw_outs, fw_outs], dim=-1)))
        outs = torch.cat([o.unsqueeze(0) for o in outs], dim=0)

        feats = self.linear1(self.relu(torch.sum(outs, dim=0)))
        feats += inps # Residual connection
        return feats

class BiGCN(nn.Module):
    def __init__(self, etypes, configs):
        super(BiGCN, self).__init__()
        self.etypes = etypes
        self.configs = configs
        self.num_hidden_layers = configs['ieg_bignn_hidden_layers']

        bigcn_layers = []
        for _ in range(self.num_hidden_layers):
            bigcn_layers.append(BiGCNLayer(etypes, configs))
        self.bigcn_layers = nn.ModuleList(bigcn_layers)

    def forward(self, embs, fw_adjs, bw_adjs):
        out = embs
        for i in range(self.num_hidden_layers):
            out = self.bigcn_layers[i](out, fw_adjs, bw_adjs)
        return out

class KnowledgeEnhancerModule(nn.Module):
    def __init__(self, configs):
        super(KnowledgeEnhancerModule, self).__init__()
        self.configs = configs
        self.cuid2embs = pickle.load(open(UMLS_EMBS, 'rb'))
        print('Size of cuid2embs: {}'.format(len(self.cuid2embs)))

        # Edge types of external knowledge graphs
        self.ekg_etypes = set()
        with open(UMLS_RELTYPES_FILE, 'r') as f:
            for line in f:
                self.ekg_etypes.add(line.strip().split('|')[1])
        self.ekg_etypes = list(self.ekg_etypes)
        self.ekg_etypes.sort()

        # RGCNModel for external knowledge graphs
        self.ekg_gnn_model = RGCNModel(self.ekg_etypes, h_dim=UMLS_EMBS_SIZE,
                                      num_bases=configs['ekg_gnn_num_bases'],
                                      num_hidden_layers=configs['ekg_gnn_hidden_layers'],
                                      dropout=configs['ekg_gnn_dropout'], use_self_loop=True)
        self.ekg_out_linear = nn.Linear(UMLS_EMBS_SIZE, configs['span_emb_size'])
        self.relu = nn.ReLU()

        # GCNs for prior IE graph
        if configs['dataset'] == ADE: nb_ieg_etypes = len(ADE_RELATION_TYPES)
        elif configs['dataset'] == BIORELEX: nb_ieg_etypes = len(BIORELEX_RELATION_TYPES)
        self.ieg_etypes = list(range(nb_ieg_etypes))
        self.bigcn = BiGCN(self.ieg_etypes, configs)

        # AttentionBasedLinkingModule
        self.linker = AttentionBasedLinkingModule(configs)

        # Logging Flags
        self.knowledge_module_logging = False
        self.log_f = None

    def start_logging(self):
        self.knowledge_module_logging = True
        if os.path.exists(KNOWLEDGE_MODULE_LOG_FILE):
            os.remove(KNOWLEDGE_MODULE_LOG_FILE)
        self.log_f = open(KNOWLEDGE_MODULE_LOG_FILE, 'w+', encoding='utf-8')

    def end_logging(self):
        self.knowledge_module_logging = False
        if self.log_f:
            self.log_f.close()
            self.log_f = None

    def forward(self, text, ie_preds):
        tokenization = ie_preds['tokenization']

        # Extract external kg and apply RGCN on it
        ekg_graph, nodes = umls_extract_network(text)
        ekg_concepts = umls_search_concepts([text])[0][0]['concepts']
        ekg_graph = ekg_graph.to(self.device)
        initial_node_embs = torch.tensor([self.cuid2embs[n] for n in nodes]).to(self.device)
        ekg_in_h = {NODE: initial_node_embs}
        ekg_out_h = self.ekg_gnn_model(ekg_graph, ekg_in_h)[NODE]
        ekg_out_h = self.relu(self.ekg_out_linear(ekg_out_h))

        # Process prior IE predictions
        candidate_starts, candidate_ends = tolist(ie_preds['starts']), tolist(ie_preds['ends'])
        candidate_char_starts = [tokenization['token2startchar'][s] for s in candidate_starts]
        candidate_char_ends = [tokenization['token2endchar'][e] for e in candidate_ends]
        candidate_spans = list(zip(candidate_char_starts, candidate_char_ends))
        candidate_embs, relation_probs = ie_preds['embs'], ie_preds['relation_probs']
        fw_adjs, bw_adjs = self.adjs_from_preds(relation_probs)

        # Apply BiGCN on the adjacency matrices
        ieg_out_h = self.bigcn(candidate_embs, fw_adjs, bw_adjs)

        # Construct span2concepts
        span2concepts = []
        for ix, span_loc in enumerate(candidate_spans):
            indexes = []
            for concept in ekg_concepts:
                concept_cui = concept['cui']
                concept_loc = concept['start_char'], concept['end_char']
                if span_loc == concept_loc and concept_cui in nodes:
                    indexes.append(nodes.index(concept_cui))
            span2concepts.append(indexes)

        # Apply AttentionBasedLinkingModule
        ka_span_embs, attention_probs = self.linker(ieg_out_h, ekg_out_h, span2concepts)
        ka_span_embs = torch.cat([ka_span_embs, ieg_out_h], dim=-1)

        # Logging
        if self.knowledge_module_logging:
            # Build nodeidx2info
            nodeidx2info = {}
            for jx, node in enumerate(nodes):
                node_semtypes, node_texts, appeared_cuis = [], [], set()
                for concept in ekg_concepts:
                    concept_cui = concept['cui']
                    concept_semtypes = concept['semtypes']
                    concept_loc = concept['start_char'], concept['end_char']
                    if concept_cui in appeared_cuis: continue
                    if concept_cui == node:
                        node_semtypes += concept_semtypes
                        node_texts.append(text[concept_loc[0]:concept_loc[1]])
                    appeared_cuis.add(concept_cui)
                nodeidx2info[jx] = {
                    'text': list(set(node_texts)),
                    'semtypes': list(set(node_semtypes)),
                    'cui': node
                }

            # Logging
            all_candidates = [text[s[0]:s[1]] for s in candidate_spans]
            for ix, span_loc in enumerate(candidate_spans):
                if len(span2concepts[ix]) > 0:
                    self.log_f.write('\ntext = {}\n'.format(text))
                    self.log_f.write('all_candidates = {}\n'.format(all_candidates))
                    self.log_f.write('candidate text = {}\n'.format(text[span_loc[0]: span_loc[1]]))
                    for jx in span2concepts[ix]:
                        self.log_f.write('c = {}\n'.format(nodeidx2info[jx]))
                    self.log_f.write('tensor = {}\n'.format(attention_probs[ix]))
        return ka_span_embs

    def adjs_from_preds(self, relation_probs):
        relation_probs = relation_probs.clone().detach()
        fw_adjs, bw_adjs = [], []
        nb_nodes = relation_probs.size()[0]
        for ix in range(len(self.ieg_etypes)):
            A = relation_probs[:,:,ix]
            # Fill diagonal with zero
            A.fill_diagonal_(0)
            # Update fw_adjs and bw_adjs
            fw_adjs.append(A.to(self.device))
            bw_adjs.append(A.T.to(self.device))
        return fw_adjs, bw_adjs
