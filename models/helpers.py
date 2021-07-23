import torch
from constants import *

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.
    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def texts_from_locs(data, token_starts, token_ends):
    mention_texts = []
    token2startchar = data.tokenization['token2startchar']
    token2endchar = data.tokenization['token2endchar']
    for s, e in zip(token_starts, token_ends):
        start_char, end_char = token2startchar[int(s)], token2endchar[int(e)]
        text = data.text[start_char: end_char]
        mention_texts.append(text)
    return mention_texts

# Filtering out candidates that are obviously not entity mentions to reduce the training/inference time.
def filter_mentions(data, candidate_mentions, dataset):
    if dataset == ADE: return filter_ade_mentions(data, candidate_mentions)
    if dataset == BIORELEX: return filter_biorelex_mentions(data, candidate_mentions)
    return candidate_mentions

def filter_ade_mentions(data, candidate_mentions):
    # Extract mention texts
    c_starts = [c[0] for c in candidate_mentions]
    c_ends = [c[1] for c in candidate_mentions]
    c_texts = texts_from_locs(data, c_starts, c_ends)

    # Filtering
    filtered_candidates = []
    for ix, mention_text in enumerate(c_texts):
        pass_filter = True
        words = mention_text.split(' ')
        lower_words = [w.lower() for w in words]
        nb_words = len(words)
        # Filters
        for punc in '.,-':
            if mention_text.startswith(punc): pass_filter = False
            if mention_text.endswith(punc): pass_filter = False
        for w in ADE_FILTER_WORDSET_1:
            if w in lower_words: pass_filter = False
        for w in ADE_FILTER_WORDSET_2:
            if lower_words[0] == w: pass_filter = False
            if lower_words[-1] == w: pass_filter = False
        if '(' in mention_text and (not ')' in mention_text): pass_filter = False
        if ')' in mention_text and (not '(' in mention_text): pass_filter = False
        if '[' in mention_text and (not ']' in mention_text): pass_filter = False
        if ']' in mention_text and (not '[' in mention_text): pass_filter = False
        # Update filtered_candidates
        if pass_filter:
            filtered_candidates.append((c_starts[ix], c_ends[ix]))

    return filtered_candidates

def filter_biorelex_mentions(data, candidate_mentions):
    # Extract mention texts
    c_starts = [c[0] for c in candidate_mentions]
    c_ends = [c[1] for c in candidate_mentions]
    c_texts = texts_from_locs(data, c_starts, c_ends)

    # Filtering
    filtered_candidates = []
    for ix, mention_text in enumerate(c_texts):
        pass_filter = True
        words = mention_text.split(' ')
        nb_words = len(words)
        # Filters (based on the existence of some specific substrings)
        if nb_words > 11: pass_filter = False
        for w in BIORELEX_FILTER_WORDSET_1:
            if w in words: pass_filter = False
        for w in BIORELEX_FILTER_WORDSET_2:
            if w in mention_text.lower(): pass_filter = False
        for w in BIORELEX_FILTER_WORDSET_3:
            if w in mention_text: pass_filter = False
        if words[0].lower() in ['by','between', 'in', 'of', 'to', 'the', 'with', 'it', 'binding', 'as', 'for', 'after']:
            pass_filter = False
        if words[-1].lower() in ['by','between', 'in', 'of', 'to', 'the', 'with', 'it', 'as', 'for', 'after']:
            pass_filter = False
        # Filters (based on parentheses)
        if '(' in mention_text and (not ')' in mention_text): pass_filter = False
        if ')' in mention_text and (not '(' in mention_text): pass_filter = False
        if '[' in mention_text and (not ']' in mention_text): pass_filter = False
        if ']' in mention_text and (not '[' in mention_text): pass_filter = False
        # Filters (based on starting/ending)
        for start in ['bound', '-bound', 'and ', ')']:
            if mention_text.startswith(start): pass_filter = False
        for end in [',', '.', '_', 'the', ' to', ' and', '?', ' in', 'ble', '[', ' with', ' of', ' by', 'is ', ' is']:
            if mention_text.endswith(end): pass_filter = False
        # Update filtered_candidates
        if pass_filter:
            filtered_candidates.append((c_starts[ix], c_ends[ix]))

    return filtered_candidates
