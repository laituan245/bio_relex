import logging
import torch
import numpy as np

from tqdm import tqdm
from constants import *

def get_entity_mentions(sentence):
    typed_mentions = set()
    for cluster in sentence['entities']:
        for alias, entity in cluster['names'].items():
            for start, end in entity['mentions']:
                if entity['is_mentioned']:
                    typed_mentions.add((start, end, cluster['label']))
    return typed_mentions

def get_relation_mentions(sent):
    relations = []
    for interaction in sent['interactions']:
        ta, tb = interaction['participants']
        ta, tb = sent['entities'][ta], sent['entities'][tb]
        for a_obj in ta['names'].values():
            for b_obj in tb['names'].values():
                if (not a_obj['is_mentioned']) or (not b_obj['is_mentioned']): continue
                #assert(len(a_obj['mentions']) == 1)
                #assert(len(b_obj['mentions']) == 1)
                loc_a = '_'.join([str(s) for s in a_obj['mentions'][0]])
                loc_b = '_'.join([str(s) for s in b_obj['mentions'][0]])
                rel = {
                    'head': loc_a, 'head_type': ta['label'],
                    'tail': loc_b, 'tail_type': tb['label'],
                    'type': interaction['label']
                }
                relations.append(rel)
    return relations

def evaluate_ade(model, dataset):
    num_docs = len(dataset)
    truth_sentences, pred_sentences, keys = {}, {}, set()
    with torch.no_grad():
        for i in range(num_docs):
            truth_sentences[dataset[i].id] = dataset[i].data
            pred_sentences[dataset[i].id] = model.predict(dataset[i])
            keys.add(dataset[i].id)

    # Evaluate NER
    gt_entities, pred_entities, entity_types = [], [], set()
    for key in keys:
        typed_truth = get_entity_mentions(truth_sentences[key])
        typed_pred = get_entity_mentions(pred_sentences[key])
        # Update gt_entities
        batch_gt_entities = []
        for start, end, type in typed_truth:
            if not type == NOT_ENTITY: entity_types.add(type)
            batch_gt_entities.append({
                'start': start, 'end': end, 'type': type
            })
        gt_entities.append(batch_gt_entities)
        # Update pred_entities
        batch_pred_entities = []
        for start, end, type in typed_pred:
            if not type == NOT_ENTITY: entity_types.add(type)
            batch_pred_entities.append({
                'start': start, 'end': end, 'type': type
            })
        pred_entities.append(batch_pred_entities)
    entity_types = list(entity_types)
    entity_score = ner_score(pred_entities, gt_entities, entity_types)

    # Evaluate RE
    pred_relations, gt_relations, relation_types = [], [], set()
    for key in keys:
        # Update gt_relations
        typed_truth = get_relation_mentions(truth_sentences[key])
        gt_relations.append(typed_truth)
        # Update pred_relations
        typed_pred = get_relation_mentions(pred_sentences[key])
        pred_relations.append(typed_pred)
        # Update relation_types
        for rel in typed_truth:
            if rel['type'] == NOT_RELATION: continue
            relation_types.add(rel['type'])
        for rel in typed_pred:
            if rel['type'] == NOT_RELATION: continue
            relation_types.add(rel['type'])
    relation_types = list(relation_types)
    relation_score = re_score(pred_relations, gt_relations, relation_types)

    # Macro scores
    m_score = entity_score['ALL']['Macro_f1']
    relex_all_score = relation_score['ALL']['Macro_f1']
    print('Mention Score (F1) = {} | Relation Extraction (F1) = {}'.format(m_score, relex_all_score))
    return m_score, relex_all_score

def ner_score(pred_entities, gt_entities, entity_types):
    """Evaluate NER predictions
    Args:
        pred_entities (list) :  list of list of predicted entities (several entities in each sentence)
        gt_entities (list) :    list of list of ground truth entities
            entity = {"start": start_idx (inclusive),
                      "end": end_idx (exclusive),
                      "type": ent_type}
        entity_types :       Entity Types """
    assert len(pred_entities) == len(gt_entities)

    scores = {ent: {"tp": 0, "fp": 0, "fn": 0} for ent in entity_types + ["ALL"]}

    # Count GT entities and Predicted entities
    n_sents = len(gt_entities)
    n_phrases = sum([len([ent for ent in sent]) for sent in gt_entities])
    n_found = sum([len([ent for ent in sent]) for sent in pred_entities])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_entities, gt_entities):
        for ent_type in entity_types:
            pred_ents = {(ent["start"], ent["end"]) for ent in pred_sent if ent["type"] == ent_type}
            gt_ents = {(ent["start"], ent["end"]) for ent in gt_sent if ent["type"] == ent_type}

            scores[ent_type]["tp"] += len(pred_ents & gt_ents)
            scores[ent_type]["fp"] += len(pred_ents - gt_ents)
            scores[ent_type]["fn"] += len(gt_ents - pred_ents)

    # Compute per entity Precision / Recall / F1
    for ent_type in scores.keys():
        if scores[ent_type]["tp"]:
            scores[ent_type]["p"] = 100 * scores[ent_type]["tp"] / (scores[ent_type]["fp"] + scores[ent_type]["tp"])
            scores[ent_type]["r"] = 100 * scores[ent_type]["tp"] / (scores[ent_type]["fn"] + scores[ent_type]["tp"])
        else:
            scores[ent_type]["p"], scores[ent_type]["r"] = 0, 0

        if not scores[ent_type]["p"] + scores[ent_type]["r"] == 0:
            scores[ent_type]["f1"] = 2 * scores[ent_type]["p"] * scores[ent_type]["r"] / (
                    scores[ent_type]["p"] + scores[ent_type]["r"])
        else:
            scores[ent_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[ent_type]["tp"] for ent_type in entity_types])
    fp = sum([scores[ent_type]["fp"] for ent_type in entity_types])
    fn = sum([scores[ent_type]["fn"] for ent_type in entity_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in entity_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in entity_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in entity_types])

    logging.info(
        "processed {} sentences with {} phrases; found: {} phrases; correct: {}.".format(n_sents, n_phrases, n_found,
                                                                                         tp))
    logging.info(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    logging.info(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    logging.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for ent_type in entity_types:
        logging.info("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            ent_type,
            scores[ent_type]["tp"],
            scores[ent_type]["fp"],
            scores[ent_type]["fn"],
            scores[ent_type]["p"],
            scores[ent_type]["r"],
            scores[ent_type]["f1"],
            scores[ent_type]["tp"] +
            scores[ent_type][
                "fp"]))

    return scores


def re_score(pred_relations, gt_relations, relation_types, mode="strict"):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}
        relation_types :        Relation Types
        mode (str) :            in 'strict' or 'boundaries' """

    assert mode in ["strict", "boundaries"]

    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                             rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                           rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per entity Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

    logging.info(f"RE Evaluation in *** {mode.upper()} *** mode")

    logging.info(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,
                                                                                             tp))
    logging.info(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    logging.info(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    logging.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        logging.info("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            rel_type,
            scores[rel_type]["tp"],
            scores[rel_type]["fp"],
            scores[rel_type]["fn"],
            scores[rel_type]["p"],
            scores[rel_type]["r"],
            scores[rel_type]["f1"],
            scores[rel_type]["tp"] +
            scores[rel_type][
                "fp"]))

    return scores
