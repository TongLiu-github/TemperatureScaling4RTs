import torch
import argparse
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax
from tqdm import tqdm
import glob
import sys
from statistics import mean


def surprisal_calculate(probs, true_class):
    return  -1 * torch.log(torch.tensor(probs[true_class]))

@torch.no_grad()
def get_surprisal_per_article(model, input_y_ids_list, target_span_list, target_len_list, loss_fct, device, batchsize, num_pieces_added, tokenizer):

    # num_pieces_added: the number of pieces in this article (when there are two tokens, count them as two).
    random.seed(0)
    pad_id = -100
    
    assert len(input_y_ids_list) == len(target_span_list)
    assert len(target_span_list) == len(target_len_list)
    inputids_target_spans_idx = [
        (_text, _target_span, idx) for idx, (_text, _target_span) in enumerate(zip(input_y_ids_list, target_span_list))
    ]  
    # sort according to the length of input ids, so the predicitons of 2 grams are in the front. 
    inputids_target_spans_idx = sorted(inputids_target_spans_idx, key=lambda x: len(x[0]))

    sorted_input_y = list(map(lambda x: x[0], inputids_target_spans_idx))
    sorted_target_spans = list(map(lambda x: x[1], inputids_target_spans_idx))
    sorted_idxs = list(map(lambda x: x[2], inputids_target_spans_idx))
    
    sorted_input_ids = [ids[:-1] for ids in sorted_input_y]
    sorted_gold_ids = [
        ids[1:] for ids, span in zip(sorted_input_y, sorted_target_spans)
    ]
    assert len(sorted_input_ids) == len(sorted_gold_ids)
    
    # here is a unordered list 
    surprisals_list = []
    surprisals_list_with_order = [0] * len(sorted_idxs)
    
    pred_test_logits = []
    pred_test_probs = []
    pred_test_labels = []
    true_test_labels = []
    
    logits_array_with_order = torch.zeros((num_pieces_added, tokenizer.vocab_size))
    probs_array_with_order = torch.zeros((num_pieces_added, tokenizer.vocab_size))
    true_labels_list_with_order = [0] * len(sorted_idxs)
    
    memory_sample_idx = []
    
    for i in range(math.ceil(len(input_y_ids_list) / batchsize)):
        batched_input_ids = sorted_input_ids[batchsize * i : batchsize * (i + 1)]
        batched_gold_ids = sorted_gold_ids[batchsize * i : batchsize * (i + 1)]
        batched_target_spans = sorted_target_spans[batchsize * i : batchsize * (i + 1)]
        batched_idxs = sorted_idxs[batchsize * i : batchsize * (i + 1)]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            batched_input_ids, batch_first=True, padding_value=pad_id
        )
        padded_gold_ids = torch.nn.utils.rnn.pad_sequence(
            batched_gold_ids, batch_first=True, padding_value=pad_id
        )
        input_mask = (padded_input_ids > -1).int()
        padded_input_ids = torch.where(padded_input_ids == pad_id, 0, padded_input_ids)
        padded_gold_ids = torch.where(padded_gold_ids == pad_id, 0, padded_gold_ids)
        assert (padded_input_ids > -1).all()
        assert len(padded_input_ids) == len(padded_gold_ids)

        # SHAPE: (batchsize, max_length, vocab_size)
        results = model(
            input_ids=padded_input_ids.to(device), attention_mask=input_mask.to(device)
        )["logits"]
        
        
        

        for next_token_score, _target_span, padded_gold_id, _sample_idx in zip(
            results, batched_target_spans, padded_gold_ids, batched_idxs
        ):
            _tartget_len = _target_span[1] - _target_span[0] + 1
            
            if _tartget_len == 1:
                _scores = next_token_score[_target_span[0] - 1, :]
                pred_test_logits.append(_scores.cpu().detach())
                _probs = _scores.softmax(-1)
                pred_test_probs.append(_probs.cpu().detach())
                pred_test_labels.append(torch.argmax(_probs).cpu().detach())
                _true_classes = padded_gold_id[_target_span[0] - 1]
                true_test_labels.append(_true_classes)
                _surprisal = surprisal_calculate(_probs, _true_classes)
                _surprisal = _surprisal.item()
                surprisals_list.append(_surprisal)
                surprisals_list_with_order[_sample_idx] = _surprisal
        
                true_labels_list_with_order[_sample_idx] = _true_classes.item()
                logits_array_with_order[_sample_idx, :] = _scores.cpu().detach()
                probs_array_with_order[_sample_idx, :] = _probs.cpu().detach()
                
                memory_sample_idx.append((_sample_idx, _tartget_len))
                
            elif _tartget_len > 1:
                
                _scores = next_token_score[_target_span[0] - 1 : _target_span[1]]
                _probs = _scores.softmax(-1)
                _true_classes = padded_gold_id[_target_span[0] - 1 : _target_span[1]]
                
                for __scores, __probs, __true_class in zip(_scores, _probs, _true_classes):
                    pred_test_probs.append(__probs.cpu().detach())
                    true_test_labels.append(__true_class)
                    pred_test_logits.append(__scores.cpu().detach())
                    pred_test_labels.append(torch.argmax(__probs).cpu().detach())
                    
                    true_labels_list_with_order[_sample_idx] = __true_class.item()
                    logits_array_with_order[_sample_idx, :] = __scores.cpu().detach()
                    probs_array_with_order[_sample_idx, :] = __probs.cpu().detach()
                          
                    memory_sample_idx.append((_sample_idx, _tartget_len))
                
                _surprisal_list = [surprisal_calculate(__probs, __true_class).item() for __probs, __true_class in zip(_probs, _true_classes)]
                surprisals_list.extend(_surprisal_list)
                _surprisal = sum(_surprisal_list)
                surprisals_list_with_order[_sample_idx] = _surprisal
                

            else:
                print("error")

    # stack
    pred_test_probs = torch.stack(pred_test_probs).numpy()
    true_test_labels = torch.stack(true_test_labels).numpy()
    pred_test_logits = torch.stack(pred_test_logits).numpy()
    pred_test_labels = torch.stack(pred_test_labels).numpy() 
    
    
    return (surprisals_list_with_order, surprisals_list), \
            (pred_test_logits, pred_test_probs, pred_test_labels, true_test_labels), \
            (logits_array_with_order, probs_array_with_order, true_labels_list_with_order), \
            memory_sample_idx



