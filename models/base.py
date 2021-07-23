import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random

from transformers import *
from math import ceil, floor

# Optimizer
class ModelOptimizer(object):
    def __init__(self, transformer_optimizer, transformer_scheduler,
                 task_optimizer, task_init_lr, max_iter):
        self.iter = 0
        self.transformer_optimizer = transformer_optimizer
        self.transformer_scheduler = transformer_scheduler

        self.task_optimizer = task_optimizer
        self.task_init_lr = task_init_lr
        self.max_iter = max_iter

    def zero_grad(self):
        self.transformer_optimizer.zero_grad()
        self.task_optimizer.zero_grad()

    def step(self):
        self.iter += 1
        self.transformer_optimizer.step()
        self.task_optimizer.step()
        self.transformer_scheduler.step()
        self.poly_lr_scheduler(self.task_optimizer, self.task_init_lr, self.iter, self.max_iter)

    @staticmethod
    def poly_lr_scheduler(optimizer, init_lr, iter, max_iter,
                          lr_decay_iter=1, power=1.0):
        """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param max_iter is number of maximum iterations
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param power is a polymomial power
        """
        if iter % lr_decay_iter or iter > max_iter:
            return optimizer

        lr = init_lr*(1 - iter/max_iter)**power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

# BaseModel
class BaseModel(nn.Module):
    def __init__(self, configs):
        super(BaseModel, self).__init__()
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() and not configs['no_cuda'] else 'cpu')

    def get_optimizer(self, num_warmup_steps, num_train_steps, start_iter = 0):
        # Extract transformer parameters and task-specific parameters
        transformer_params, task_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "transformer.encoder" in name:
                    transformer_params.append((name, param))
                else:
                    task_params.append((name, param))

        # Prepare transformer_optimizer and transformer_scheduler
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in transformer_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in transformer_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        transformer_optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.configs['transformer_learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-06,
        )
        transformer_scheduler = get_linear_schedule_with_warmup(transformer_optimizer,
                                                                num_warmup_steps=num_warmup_steps,
                                                                num_training_steps=num_train_steps)

        # Prepare the optimizer for task-specific parameters
        task_optimizer = optim.Adam([p for n, p in task_params], lr=self.configs['task_learning_rate'])

        # Unify transformer_optimizer and task_optimizer
        model_optimizer = ModelOptimizer(transformer_optimizer, transformer_scheduler,
                                         task_optimizer, self.configs['task_learning_rate'],
                                         num_train_steps)
        model_optimizer.iter = start_iter

        return model_optimizer

# FFNN Module
class FFNNModule(nn.Module):
    """ Generic FFNN-based Scoring Module
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout = 0.2):
        super(FFNNModule, self).__init__()
        self.layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.squeeze()

# TransformerForElementScorer
class TransformerForElementScorer(nn.Module):
    def __init__(self, d_model, n_heads, d_feedforward, n_layers, d_output=1):
        super(TransformerForElementScorer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_feedforward)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)
        self.output_linear = nn.Linear(d_model, d_output)

    def forward(self, x):
        # x must have shape [batch_size, seq_len, d_model]
        x = x.transpose(1, 0)
        features = self.transformer(x)
        features = features.transpose(1, 0)
        scores = self.output_linear(features)
        return scores, features
