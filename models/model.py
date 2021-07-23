import torch
import utils
import random
import numpy as np

from utils import *
from constants import *
from models.base import *
from models.encoder import *
from models.helpers import *
from models.external_knowledge import *

THRESHOLD = 0.50
OUTPUT_FIELDS = ['starts', 'ends', 'entity_labels', 'relation_labels']

# Main Class
class PredictionHead(nn.Module):
    def __init__(self, configs, device, final_head=True, typed=True,
                 knowledge_enhanced=False):
        super(PredictionHead, self).__init__()
        self.configs = configs
        self.typed = typed
        self.device = device
        self.span_emb_size = configs['span_emb_size']
        self.pair_embs_size = 3 * self.span_emb_size
        self.nb_entity_types = len(configs['entity_types']) if typed else 2
        self.nb_relation_types = len(configs['relation_types']) if typed else 2
        self.final_head = final_head

        # Mention Scorer
        mention_hidden_sizes = [configs['mention_scorer_ffnn_size']] * configs['mention_scorer_ffnn_depth']
        mention_scorer_input_size = self.span_emb_size
        if knowledge_enhanced: mention_scorer_input_size += self.span_emb_size
        self.mention_scorer = FFNNModule(input_size = mention_scorer_input_size,
                                         hidden_sizes = mention_hidden_sizes,
                                         output_size = self.nb_entity_types,
                                         dropout = configs['dropout_rate'])
        self.mention_loss_fct = nn.CrossEntropyLoss()


        # Relation Extractor
        relation_hidden_sizes = [configs['mention_linker_ffnn_size']] * configs['mention_linker_ffnn_depth']
        relation_scorer_input_size = self.pair_embs_size
        if knowledge_enhanced: relation_scorer_input_size += self.pair_embs_size
        self.relation_scorer = FFNNModule(input_size = relation_scorer_input_size,
                                          hidden_sizes = relation_hidden_sizes,
                                          output_size = self.nb_relation_types,
                                          dropout = configs['dropout_rate'])
        self.relation_loss_fct = nn.CrossEntropyLoss()

    def forward(self, num_tokens, candidate_starts, candidate_ends, candidate_embs,
                mention_labels, relation_labels, is_training, in_ned_pretraining):
        if not self.typed:
            mention_labels = (mention_labels > 0).type(torch.LongTensor).to(self.device)
            relation_labels = (relation_labels > 0).type(torch.LongTensor).to(self.device)

        # Compute mention_scores
        mention_scores = self.mention_scorer(candidate_embs)
        if len(mention_scores.size()) == 1: mention_scores = mention_scores.unsqueeze(0)
        mention_loss = self.mention_loss_fct(mention_scores, mention_labels) if is_training else 0
        _, pred_mention_labels = torch.max(mention_scores, 1)
        if is_training and in_ned_pretraining: return mention_loss, {l:[] for l in OUTPUT_FIELDS}

        # Extract top candidates
        if self.final_head:
            # Extract candidates that do not have not-entity label
            top_candidate_indexes = [ix for ix, l in enumerate(tolist(pred_mention_labels)) if l > 0]
            if len(top_candidate_indexes) == 0:
                nb_mentions = len(candidate_starts)
                top_candidate_indexes = random.sample(list(range(nb_mentions)), 1)
        else:
            # Extract candidates with low not-entity scores
            not_entity_scores = torch.softmax(mention_scores, dim=-1)[:,0]
            c = int(max(num_tokens * self.configs['span_ratio'], 1))
            _, top_candidate_indexes = torch.topk(-not_entity_scores, k=c)
            top_candidate_indexes = tolist(torch.sort(top_candidate_indexes)[0])

        top_candidate_indexes = torch.tensor(top_candidate_indexes).to(self.device)
        top_candidate_starts = candidate_starts[top_candidate_indexes]
        top_candidate_ends = candidate_ends[top_candidate_indexes]
        top_mention_scores = torch.index_select(mention_scores, 0, top_candidate_indexes)
        top_mention_labels = pred_mention_labels[top_candidate_indexes]
        top_candidate_embs = torch.index_select(candidate_embs, 0, top_candidate_indexes)
        relation_labels = torch.index_select(relation_labels, 0, top_candidate_indexes)
        relation_labels = torch.index_select(relation_labels, 1, top_candidate_indexes)

        # Compute pair_embs
        pair_embs = self.get_pair_embs(top_candidate_embs)

        # Compute pair_relation_scores and pair_relation_loss
        pair_relation_scores = self.relation_scorer(pair_embs)
        if len(pair_relation_scores.size()) <= 1:
            pair_relation_scores = pair_relation_scores.view(1, 1, self.nb_relation_types)
        pair_relation_labels = torch.max(pair_relation_scores, 2)[1]
        pair_relation_loss = 0
        if is_training:
            pair_relation_loss = self.relation_loss_fct(pair_relation_scores.view(-1, self.nb_relation_types), relation_labels.view(-1))

        # Compute total_loss
        total_loss = mention_loss + pair_relation_loss

        # Compute probs
        top_mention_probs = torch.softmax(top_mention_scores, dim=-1)
        pair_relation_probs = torch.softmax(pair_relation_scores, dim=-1)

        # Build preds
        preds = {
            'starts': top_candidate_starts, 'ends': top_candidate_ends, 'embs': top_candidate_embs,
            'entity_labels': top_mention_labels, 'relation_labels': pair_relation_labels,
            'entity_probs': top_mention_probs, 'relation_probs': pair_relation_probs
        }
        return total_loss, preds

    def get_pair_embs(self, candidate_embs):
        n, d = candidate_embs.size()
        features_list = []

        # Compute diff_embs and prod_embs
        src_embs = candidate_embs.view(1, n, d).repeat([n, 1, 1])
        target_embs = candidate_embs.view(n, 1, d).repeat([1, n, 1])
        prod_embds = src_embs * target_embs

        # Update features_list
        features_list.append(src_embs)
        features_list.append(target_embs)
        features_list.append(prod_embds)

        # Concatenation
        pair_embs = torch.cat(features_list, 2)

        return pair_embs

class JointModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.configs = configs
        self.nb_entity_types = len(configs['entity_types'])
        self.nb_relation_types = len(configs['relation_types'])
        self.in_ned_pretraining = False
        self.use_external_knowledge = self.configs['use_external_knowledge']

        # Transformer Encoder
        self.encoder = TransformerEncoder(configs)

        # Span Embedding Linear
        self.span_linear_1 = nn.Linear(3 * self.encoder.hidden_size + self.configs['feature_size'], 1000)
        self.span_linear_2 = nn.Linear(1000, self.configs['span_emb_size'])
        self.span_relu = nn.ReLU()

        # Head-finding attention (If enabled)
        self.attention_scorer = FFNNModule(input_size = self.encoder.hidden_size,
                                           hidden_sizes = [],
                                           output_size = 1,
                                           dropout = configs['dropout_rate'])

        # Span-width features
        self.span_width_embeddings = nn.Embedding(configs['max_span_width'], configs['feature_size'])

        # Knowledge-Enhancer Module (if enabled)
        if self.use_external_knowledge:
            self.knowledge_enhancer = KnowledgeEnhancerModule(configs)
            self.knowledge_enhancer.device = self.device
            self.knowledge_enhancer.linker.device = self.device

        # Prediction Heads
        self.predictor1 = PredictionHead(configs, self.device, final_head = not self.use_external_knowledge)
        if self.use_external_knowledge:
            self.predictor2 = PredictionHead(configs, self.device, knowledge_enhanced = True)

        # Move to Device
        self.to(self.device)

    def get_span_emb_size(self):
        return self.configs['span_emb_size']

    def forward(self, input_ids, input_masks, mask_windows,
                gold_starts, gold_ends, gold_labels, isstartingtoken,
                cluster_ids, relations, data, is_training):
        self.train() if is_training else self.eval()

        # Transformer Encoder
        num_windows, window_size = input_ids.size()[:2]
        transformer_features, pooler_features = \
            self.encoder(input_ids, input_masks, mask_windows, num_windows, window_size, is_training)
        num_tokens = transformer_features.size()[0]

        # Enumerate span candidates
        candidate_spans = self.enumerate_candidate_spans(num_tokens, self.configs['max_span_width'], isstartingtoken)
        candidate_spans = sorted(candidate_spans, key=lambda x: x[0])
        candidate_starts = torch.LongTensor([s[0] for s in candidate_spans]).to(self.device)
        candidate_ends = torch.LongTensor([s[1] for s in candidate_spans]).to(self.device)

        # Extract candidate embeddings
        candidate_embs = self.get_span_emb(transformer_features, candidate_starts, candidate_ends)
        candidate_embs = self.span_relu(self.span_linear_1(candidate_embs))
        candidate_embs = self.span_relu(self.span_linear_2(candidate_embs))

        # Apply PredictionHead (1st)
        mention_labels = self.get_mention_labels(candidate_spans, gold_starts, gold_ends, gold_labels).to(self.device)
        relation_labels = self.get_relation_labels(candidate_starts, candidate_ends, relations)
        loss1, preds = \
            self.predictor1(num_tokens, candidate_starts, candidate_ends, candidate_embs,
                            mention_labels, relation_labels, is_training, self.in_ned_pretraining)
        if is_training and self.in_ned_pretraining: return loss1, [preds[l] for l in OUTPUT_FIELDS]

        # if use_external_knowledge
        loss2 = 0.0
        if self.configs['use_external_knowledge']:
            # Knowledge-aware span embeddings
            preds['tokenization'] = data.tokenization
            ka_span_embs = self.knowledge_enhancer(data.text, preds)
            # Apply predictor2
            candidate_starts = preds['starts']
            candidate_ends = preds['ends']
            candidate_embs = ka_span_embs
            candidate_spans = [(s, e) for (s, e) in zip(tolist(candidate_starts), tolist(candidate_ends))]
            mention_labels = self.get_mention_labels(candidate_spans, gold_starts, gold_ends, gold_labels).to(self.device)
            relation_labels = self.get_relation_labels(candidate_starts, candidate_ends, relations)
            loss2, preds = \
                self.predictor2(num_tokens, candidate_starts, candidate_ends, candidate_embs,
                                mention_labels, relation_labels, is_training, self.in_ned_pretraining)

        # total loss
        loss = loss1 + 2 * loss2

        return loss, [preds[l] for l in OUTPUT_FIELDS]

    def predict(self, instance):
        self.eval()

        # Apply the model
        tensorized_example = [b.to(self.device) for b in instance.example]
        tensorized_example.append(instance.all_relations)
        tensorized_example.append(instance)
        tensorized_example.append(False) # is_training
        preds = self.forward(*tensorized_example)[1]
        preds = [x.cpu().data.numpy() for x in preds]
        mention_starts, mention_ends, mention_labels, pair_relation_labels = preds
        nb_mentions = len(mention_labels)

        # Build loc2label
        loc2label = {}
        for i in range(nb_mentions):
            loc2label[(mention_starts[i], mention_ends[i])] = mention_labels[i]

        # Initialize sample to be returned
        interactions, entities = [], []
        sample = {
            'id': instance.id, 'text': instance.text,
            'interactions': interactions, 'entities': entities
        }

        # Get clusters
        predicted_clusters, mention_to_predicted = [], {}
        for m_start, m_end in zip(mention_starts, mention_ends):
            if not (m_start, m_end) in mention_to_predicted:
                singleton_cluster = [(m_start, m_end)]
                predicted_clusters.append(singleton_cluster)
                mention_to_predicted[(m_start, m_end)] = singleton_cluster

        # Populate entities
        mention2entityid = {}
        for entityid, cluster in enumerate(predicted_clusters):
            mentions, entity_labels = [], []
            for start_token, end_token in cluster:
                start_char = instance.tokenization['token2startchar'][start_token]
                end_char = instance.tokenization['token2endchar'][end_token]
                mentions.append((start_char, end_char))
                entity_labels.append(loc2label[(start_token, end_token)])
                mention2entityid[(start_token, end_token)] = entityid
            fstart, fend = mentions[0]
            entity_name = instance.text[fstart:fend]
            entity_label = self.configs['entity_types'][find_majority(entity_labels)]
            if entity_label != NOT_ENTITY:
                entities.append({
                    'label': entity_label,
                    'names': {
                         entity_name: {
                             'is_mentioned': True,
                             'mentions': mentions,
                             'label': entity_label,
                         }
                     },
                     'is_mentioned': True
                })

        # Populate interactions
        pred_interactions = {}
        for i in range(nb_mentions):
            start_idx = i if self.configs['symmetric_relation'] else 0
            for j in range(start_idx, nb_mentions):
                loci = mention_starts[i], mention_ends[i]
                entityi = mention2entityid[loci]
                locj = mention_starts[j], mention_ends[j]
                entityj = mention2entityid[locj]
                if not (entityi, entityj) in pred_interactions:
                    pred_interactions[(entityi, entityj)] = []
                pred_interactions[(entityi, entityj)].append(pair_relation_labels[i, j])

        for (a_idx, b_idx) in pred_interactions:
            label = find_majority(pred_interactions[(a_idx, b_idx)])
            if label == 0 or len(entities) == 0: continue
            interactions.append({
                'participants': [a_idx, b_idx],
                'label': self.configs['relation_types'][label]
            })

        return sample

    def enumerate_candidate_spans(self, num_tokens, max_span_width, isstartingtoken):
        # Generate candidate spans
        candidate_spans = set([])
        for i in range(num_tokens):
            if isstartingtoken[i] == 0: continue
            for j in range(i, i+max_span_width):
                if j >= num_tokens: continue
                if (j == num_tokens-1) or isstartingtoken[j+1] == 1:
                    candidate_spans.add((i, j))

        return list(candidate_spans)

    def get_mention_labels(self, candidate_spans, gold_starts, gold_ends, gold_labels):
        gold_starts = gold_starts.cpu().data.numpy().tolist()
        gold_ends = gold_ends.cpu().data.numpy().tolist()
        gold_spans = list(zip(gold_starts, gold_ends))
        labels = [0] * len(candidate_spans)
        for idx, (c_start, c_end) in enumerate(candidate_spans):
            if (c_start, c_end) in gold_spans:
                g_index = gold_labels[gold_spans.index((c_start, c_end))]
                labels[idx] = g_index
        labels = torch.LongTensor(labels)
        return labels

    def get_relation_labels(self, candidate_starts, candidate_ends, relations):
        candidate_starts = candidate_starts.cpu().data.numpy().tolist()
        candidate_ends = candidate_ends.cpu().data.numpy().tolist()
        k = len(candidate_starts)
        labels = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                loc1 = candidate_starts[i], candidate_ends[i]
                loc2 = candidate_starts[j], candidate_ends[j]
                if (loc1, loc2) in relations:
                    labels[i,j] = relations[(loc1, loc2)]
                    assert(labels[i,j] > 0)
        return torch.LongTensor(labels).to(self.device)

    def get_span_emb(self, context_outputs, span_starts, span_ends):
        span_emb_list = []
        num_tokens = context_outputs.size()[0]
        span_width = span_ends - span_starts + 1

        # Extract the boundary representations for the candidate spans
        span_start_emb = torch.index_select(context_outputs, 0, span_starts)
        span_end_emb = torch.index_select(context_outputs, 0, span_ends)
        assert(span_start_emb.size()[0] == span_end_emb.size()[0])
        span_emb_list.append(span_start_emb)
        span_emb_list.append(span_end_emb)

        # Extract attention-based representations
        doc_range = torch.arange(0, num_tokens).to(next(self.attention_scorer.parameters()).device)
        range_cond_1 = span_starts.unsqueeze(1) <= doc_range
        range_cond_2 = doc_range <= span_ends.unsqueeze(1)
        doc_range_mask = range_cond_1 & range_cond_2
        attns = self.attention_scorer(context_outputs).unsqueeze(0) + torch.log(doc_range_mask.float())
        attn_probs = torch.softmax(attns,dim=1)
        head_attn_reps = torch.matmul(attn_probs, context_outputs)
        span_emb_list.append(head_attn_reps)

        # Extract span size feature
        span_width_index = span_width - 1
        span_width_emb = self.span_width_embeddings(span_width_index)
        span_emb_list.append(span_width_emb)

        # Return
        span_emb = torch.cat(span_emb_list, dim=1)
        return span_emb
