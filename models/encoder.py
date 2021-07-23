import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

from constants import *
from transformers import *

class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super(TransformerEncoder, self).__init__()
        self.configs = configs

        # Transformer Encoder
        self.transformer = AutoModel.from_pretrained(configs['transformer'])
        self.transformer.config.gradient_checkpointing  = configs['gradient_checkpointing']
        self.transformer_hidden_size = self.transformer.config.hidden_size
        self.hidden_size = self.transformer_hidden_size

    def forward(self, input_ids, input_masks, mask_windows,
                num_windows, window_size, is_training,
                context_lengths = [0], token_type_ids = None):
        self.train() if is_training else self.eval()
        num_contexts = len(context_lengths)

        features, pooler_outputs = self.transformer(input_ids, input_masks, token_type_ids)[:2]
        features = features.view(num_contexts, num_windows, -1, self.transformer_hidden_size)

        flattened_features = []
        for i in range(num_contexts):
            _features = features[i, :, :, :]
            _features = _features[:, context_lengths[i]:, :]
            _features = _features[:, : window_size, :]
            flattened_features.append(self.flatten(_features, mask_windows))
        flattened_features = torch.cat(flattened_features)

        return flattened_features.squeeze(), pooler_outputs

    def flatten(self, features, mask_windows):
        num_windows, window_size, hidden_size = features.size()
        flattened_emb = torch.reshape(features, (num_windows * window_size, hidden_size))
        boolean_mask = mask_windows > 0
        boolean_mask = boolean_mask.view([num_windows * window_size])
        return flattened_emb[boolean_mask].unsqueeze(0)
