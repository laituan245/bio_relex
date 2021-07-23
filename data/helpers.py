from utils import *

def tokenize(tokenizer, words):
    results = {}
    results['tokens'] = []
    tokens = [tokenizer.tokenize(w) for w in words]
    startchar2token, endchar2token, token_ctx, offset = {}, {}, 0, 0
    isstartingtoken = []
    for _tokens in tokens:
        for token in _tokens:
            isstartingtoken.append(int(not token.startswith('##')))
            if token.startswith('##'): token = token[2:]
            elif token.endswith('##'): token = token[:-2]
            startchar2token[offset] = token_ctx
            endchar2token[offset + len(token)] = token_ctx
            token_ctx += 1
            offset += len(token)
            results['tokens'].append(token)
        offset += 1
    results['startchar2token'] = startchar2token
    results['token2startchar'] = inverse_mapping(startchar2token)
    results['endchar2token'] = endchar2token
    results['token2endchar'] = inverse_mapping(endchar2token)
    results['isstartingtoken'] = torch.tensor(isstartingtoken)
    results['tokenizer'] = tokenizer
    return results
