#!/usr/bin/python

"""
Original evaluation script for BioRelEx: Biomedical Relation Extraction Benchmark.
Only Python 3.5+ supported.

Copyright (c) ...
"""


import re
import json
import torch
import argparse

import numpy as np

from tqdm import tqdm

from sklearn import utils as sk_utils

from typing import Tuple, Set, Dict, Any, Union, Iterable, Callable

Hash = Union[int, str]
Number = Union[int, float]
JSON_Object = Dict[str, Any]
Mention = Tuple[str, int, int]


def hash_sentence(item: JSON_Object, match_by: str = 'id') -> Hash:
    """
    Get hash of sentence object for matching. Default is id.
    Useful for debugging and/or when id's are changed.

    :param item: Object to calculate hash for
    :param match_by: Matching criteria / method
    :return: Hash representation for the object
    """
    if match_by == 'text':
        text = item['text']
        text = text.lower()
        text = re.sub('[^0-9a-z]+', '', text)
        return hash(text)
    else:
        return str(item[match_by])


def get_sentences(data: Iterable[JSON_Object],
                  match_by: str) -> Dict[Hash, JSON_Object]:
    """
    Collect sentence objects w.r.t. matching criteria.
    :param data: Iterable of sentence objects
    :param match_by: Matching criteria / method
    :return: Dict of hash: sentence objects
    """
    return {
        hash_sentence(sentence, match_by): sentence
        for sentence in data
    }


def get_entity_mentions(sentence: JSON_Object) -> Set[Mention]:
    """
    Get all entity mentions given the sentence object.
    :param sentence: Sentence object
    :return: Tuple of entity name, inclusive start and exclusive end indices
    """
    return {
        (alias, start, end)
        for cluster in sentence['entities']
        for alias, entity in cluster['names'].items()
        for start, end in entity['mentions']
        if entity['is_mentioned']
    }


def unordered_pair(a, b):
    """
    Build unordered pair. Useful for matching pairs.
    :param a: Object
    :param b: Object
    :return: Normalized (Sorted) state of the tuple
    """
    if a > b:
        return b, a
    else:
        return a, b


def get_entity_coreferences(sentence: JSON_Object) -> Set[Tuple[str, str]]:
    """
    Get all the entity coreferences.
    :return: Tuples of unordered pairs of coreference participants
    """
    return {
        unordered_pair(a, b)
        for cluster in sentence['entities']
        for a in cluster['names']
        for b in cluster['names']
        if a != b
        and cluster['names'][a]['is_mentioned']
        and cluster['names'][b]['is_mentioned']
    }


# noinspection PyPep8Naming
class PRFScores(object):
    """
    Store and calculate Precision / Recall and F_1 scores.
    Supports namespaces (useful for different files / runs / sets).
    """
    def __init__(self, name: str):
        self.name = name

        self.TP = 0
        self.FN = 0
        self.FP = 0

        self.by_id = {}

    def store_by_id(self, id: Hash,
                    TP: int, FN: int, FP: int):
        if id not in self.by_id:
            self.by_id[id] = PRFScores(self.name)

        self.by_id[id].TP += TP
        self.by_id[id].FN += FN
        self.by_id[id].FP += FP

    def add_sets(self, id: Hash,
                 truth_set: Set[Any],
                 prediction_set: Set[Any]):
        """
        Update state of the score: store new sets
        :param id: Namespace id
        :param truth_set: Set of truth data
        :param prediction_set: Set of predictions
        """
        intersection = truth_set & prediction_set

        TP = len(intersection)
        FN = len(truth_set) - TP
        FP = len(prediction_set) - TP

        self.TP += TP
        self.FN += FN
        self.FP += FP
        self.store_by_id(id, TP, FN, FP)

    def get_scores(self) -> Dict[str, Number]:
        """
        Calculate scores w.r.t. current state
        :return: Dict of { metric name : metric value }
        """
        if self.TP + self.FP == 0:
            precision = 0
        else:
            precision = self.TP / (self.TP + self.FP)

        if self.TP + self.FN == 0:
            recall = 0
        else:
            recall = self.TP / (self.TP + self.FN)

        if precision + recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'TP': self.TP,
            'FN': self.FN,
            'FP': self.FP
        }

    def print_scores(self):
        """
        Calculate score and print the results.
        """
        print('\n')
        print(self.name)

        print('          | Pred 0 | Pred 1')
        print('   True 0 |        | {:>6}'.format(self.FP))
        print('   True 1 | {:>6} | {:>6}'.format(self.FN, self.TP))

        scores = self.get_scores()

        print('      Precision: {:>5.2f}%\n'
              '      Recall:    {:>5.2f}% \n'
              '      F-score:   {:>5.2f}%'.format(scores['precision'] * 100,
                                                  scores['recall'] * 100,
                                                  scores['f_score'] * 100))


# noinspection PyPep8Naming
class PRFScoresFlatMentions(PRFScores):
    def add_sets(self, id: Hash,
                 truth_set: Set[Mention],
                 prediction_set: Set[Mention]):
        intersection = truth_set & prediction_set
        # remove the ones which intersect with TPs
        intersection_with_truth = {
            (e, start, end) for e, start, end in truth_set
            for c_e, c_start, c_end in intersection
            if c_start < end and c_end > start and e != c_e
        }
        intersection_with_pred = {
            (e, start, end) for e, start, end in prediction_set
            for c_e, c_start, c_end in intersection
            if c_start < end and c_end > start and e != c_e
        }

        truth_set -= intersection_with_truth
        prediction_set -= intersection_with_pred

        # remove the ones that are in a larger entity
        shorts_in_truth = {
            (e1, start1, end1) for e1, start1, end1 in truth_set
            for e2, start2, end2 in truth_set
            if end2 <= start1 <= start2 and e1 != e2
        }
        shorts_in_pred = {
            (e1, start1, end1) for e1, start1, end1 in prediction_set
            for e2, start2, end2 in prediction_set
            if end2 <= start1 <= start2 and e1 != e2
        }

        truth_set -= shorts_in_truth
        prediction_set -= shorts_in_pred

        TP = len(intersection)
        FN = len(truth_set) - TP
        FP = len(prediction_set) - TP

        self.TP += TP
        self.FN += FN
        self.FP += FP

        self.store_by_id(id, TP, FN, FP)


def evaluate_sentences(truth_sentences: Dict[Hash, Dict[str, Any]],
                       pred_sentences: Dict[Hash, Dict[str, Any]],
                       keys: Iterable[Hash] = None) -> Tuple[PRFScores, PRFScores]:
    relex_any_score = PRFScores('Relation Extraction (any)')
    relex_all_score = PRFScores('Relation Extraction (all)')
    mentions_score = PRFScores('Entity Mentions')
    mentions_flat_score = PRFScoresFlatMentions('Entity Mentions (flat)')
    entities_score = PRFScores('Entities')
    coref_score = PRFScores('Entity Coreferences')

    if keys is None:
        keys = truth_sentences.keys()

    for id in keys:
        # match unique entities
        if id not in pred_sentences:
            print('No prediction for sentence with ID={}'.format(id))
            continue

        truth = truth_sentences[id]
        pred = pred_sentences[id]

        truth_entity_mentions = get_entity_mentions(truth)
        pred_entity_mentions = get_entity_mentions(pred)

        mentions_score.add_sets(id, truth_entity_mentions, pred_entity_mentions)
        mentions_flat_score.add_sets(id, truth_entity_mentions, pred_entity_mentions)

        st_entities = {entity for entity, start, end in truth_entity_mentions}
        sp_entities = {entity for entity, start, end in pred_entity_mentions}
        entities_score.add_sets(id, st_entities, sp_entities)

        st_entity_coreferences = get_entity_coreferences(truth)
        sp_entity_coreferences = get_entity_coreferences(pred)
        coref_score.add_sets(id, st_entity_coreferences, sp_entity_coreferences)

        # pred_ue_to_truth_ue = {}
        #
        # for ue, ue_obj in pred['unique_entities'].items():
        #     ue = int(ue)
        #     for ve, ve_obj in ue_obj['versions'].items():
        #         if ve in truth['entity_map']:
        #             true_ue_id = int(truth['entity_map'][ve])
        #             if ue in pred_ue_to_truth_ue and pred_ue_to_truth_ue[ue] != true_ue_id:
        #                 # another version of this entity cluster was matched to a different cluster
        #                 entity_version_mismatch += 1
        #             else:
        #                 pred_ue_to_truth_ue[ue] = true_ue_id
        #         else:
        #             # pred_ue_to_truth_ue[ue] = -ue
        #             # this version does not exist in the ground truth
        #             fp_entities += 1

        # st_unique_entities = set([int(x) for x in truth['unique_entities'].keys()])
        # sp_unique_entities = set(pred_ue_to_truth_ue.values())
        # unique_entities_score.add_sets(st_unique_entities, sp_unique_entities)

        # interactions
        predicted_pairs_with_names = {
            unordered_pair(a, b)
            for interaction in pred['interactions']
            for a, a_meta in pred['entities'][interaction['participants'][0]]['names'].items()
            for b, b_meta in pred['entities'][interaction['participants'][1]]['names'].items()
            if a_meta['is_mentioned'] and b_meta['is_mentioned']
        }
        # sometimes duplicates exist

        predicted_pairs_with_names_matched = set()

        for interaction in truth['interactions']:
            # if 'implicit' in interaction and interaction['implicit']:
            #     continue
            ta, tb = interaction['participants']
            true_pairs_with_names = {
                unordered_pair(a, b)
                for a, a_obj in truth['entities'][ta]['names'].items()
                if a_obj['is_mentioned']
                for b, b_obj in truth['entities'][tb]['names'].items()
                if b_obj['is_mentioned']
            }  # no duplicates detected

            intersection = true_pairs_with_names & predicted_pairs_with_names
            predicted_pairs_with_names_matched = predicted_pairs_with_names_matched | intersection

            true_to_add = {unordered_pair(ta, tb)}

            predicted_any_to_add = set()
            predicted_all_to_add = set()

            if intersection:
                predicted_any_to_add = true_to_add

            if len(intersection) == len(true_pairs_with_names):
                predicted_all_to_add = true_to_add

            relex_any_score.add_sets(id, true_to_add, predicted_any_to_add)
            relex_all_score.add_sets(id, true_to_add, predicted_all_to_add)

        predicted_pairs_with_names_unmatched = predicted_pairs_with_names - predicted_pairs_with_names_matched
        relex_any_score.add_sets(id, set(), predicted_pairs_with_names_unmatched)
        relex_all_score.add_sets(id, set(), predicted_pairs_with_names_unmatched)

        # TODO: check labels!

    return mentions_score, relex_all_score
    # , mentions_flat_score, entities_score,
    # coref_score, relex_any_score

def evaluate_biorelex(model, dataset):
    num_docs = len(dataset)
    truth_sentences, pred_sentences, keys = {}, {}, set()
    with torch.no_grad():
        for i in range(num_docs):
            truth_sentences[dataset[i].id] = dataset[i].data
            pred_sentences[dataset[i].id] = model.predict(dataset[i])
            keys.add(dataset[i].id)
    mentions_score, relex_all_score = evaluate_sentences(truth_sentences, pred_sentences, keys)
    m_score = mentions_score.get_scores()['f_score']
    relex_all_score = relex_all_score.get_scores()['f_score']
    print('Mention Score (F1) = {} | Relation Extraction (F1) = {}'.format(m_score, relex_all_score))
    return m_score, relex_all_score
