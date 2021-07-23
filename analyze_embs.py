import json
import pickle
import numpy as np

from constants import *
from os.path import join
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from external_knowledge import umls_search_concepts

all_sents = []

# Extract all sentences from ADE
with open('resources/ade/ade_full.json', 'r') as f:
    ade_data = json.loads(f.read())
for inst in ade_data:
    all_sents.append(' '.join(inst['tokens']))

# Extract all sentences from BioRelEx
for split in ['train', 'dev','test']:
    with open('resources/biorelex/{}.json'.format(split), 'r') as f:
        biorelex_data = json.loads(f.read())
for inst in biorelex_data:
    all_sents.append(inst['text'])

print('len(all_sents) is {}'.format(len(all_sents)))

# Extract all concepts
all_concepts, appeared = [], set()
cuid2embs = pickle.load(open(UMLS_EMBS, 'rb'))
for text in all_sents:
    concepts = umls_search_concepts([text], MM_TYPES)[0][0]['concepts']
    concepts = [c for c in concepts if c['cui'] in cuid2embs and (not c['cui'] in appeared)]
    for c in concepts: appeared.add(c['cui'])
    all_concepts += concepts
print('Total number of concepts with embs is {}'.format(len(all_concepts)))

# Divide into train/test set (given the emb, predict the semtype)
split_index = int(0.9 * len(all_concepts))
train_concepts = all_concepts[:split_index]
test_concepts = all_concepts[split_index:]
train_data = [(cuid2embs[c['cui']], MM_TYPES.index(c['semtypes'][0])) for c in train_concepts]
test_data = [(cuid2embs[c['cui']], MM_TYPES.index(c['semtypes'][0])) for c in test_concepts]

# Logistic Regression
train_X = np.array([d[0] for d in train_data])
train_Y = np.array([d[1] for d in train_data])
test_X = np.array([d[0] for d in test_data])
test_Y = np.array([d[1] for d in test_data])
print('train_X: {} | train_Y: {} | test_X: {} | test_Y: {}'.format(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape))
print('MLPClassifier')
clf = MLPClassifier(random_state=1, max_iter=10000, verbose=True).fit(train_X, train_Y)
print('Train Score: {}'.format(clf.score(train_X, train_Y)))
print('Test Score: {}'.format(clf.score(test_X, test_Y)))

# Dummy Classifiers (Most Frequent)
for strategy in ['most_frequent', 'stratified']:
    print('Dummy Classifiers ({})'.format(strategy))
    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(train_X, train_Y)
    print('Train Score: {}'.format(dummy_clf.score(train_X, train_Y)))
    print('Test Score: {}'.format(dummy_clf.score(test_X, test_Y)))
