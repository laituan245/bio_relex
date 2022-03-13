from constants import *
from scorer.ade import evaluate_ade
from scorer.biorelex import evaluate_biorelex

def evaluate(model, dataset, type):
    if type in [ADE, CHEMPROT]:
        return evaluate_ade(model, dataset)
    if type == BIORELEX:
        return evaluate_biorelex(model, dataset)
