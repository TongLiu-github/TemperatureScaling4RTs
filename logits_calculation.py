import os
import json
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from statistics import mean
from collections import defaultdict
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_surprisal_per_article, surprisal_calculate




parameter_dic = {
    "Brown": {
        "num_articles": 13,
        "dir": "./PPP_Calculation_Brown/"
    },
    "Natural_Stories": {
        "num_articles": 9,
        "dir": "./PPP_Calculation_Natural_Stories/"
    }
}

def valid_data_name(value):
    valid_values = ["Brown", "Natural_Stories", "Dundee"]
    if value not in valid_values:
        raise argparse.ArgumentTypeError(f"Invalid value for -data_name. Allowed values are {', '.join(valid_values)}.")
    return value

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main(args):
    ngram = args.n
    num_article = parameter_dic[args.data_name]["num_articles"]
    dir0 = parameter_dic[args.data_name]["dir"]
    model_name = args.model_name

    data_path = dir0 + "data/{}/processed_all_stories_dict.json".format(ngram)
    article2piece = json.load(open(data_path))

    for article_idx, pieces in tqdm(article2piece.items()):
        print(article_idx, len(pieces))

    device = torch.device("cuda:{}".format(args.cuda_num) if torch.cuda.is_available() else "cpu")

    surprisal_dic_all_article = defaultdict(lambda: defaultdict(list))
    logits_dic_all_article = defaultdict(lambda: defaultdict(np.ndarray))
    probs_dic_all_article = defaultdict(lambda: defaultdict(np.ndarray))
    true_labels_dic_all_article = defaultdict(lambda: defaultdict(list))
    memory_sample_idx_all_article = defaultdict(lambda: defaultdict(list))

    tokenizer = AutoTokenizer.from_pretrained(model_name.replace("_", "-"))
    model = AutoModelForCausalLM.from_pretrained(model_name.replace("_", "-"), return_dict_in_generate=True)
    model.to(device).eval()

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")

    surprisals_list_all_article = []

    # calculate

    pred_test_logits, pred_test_probs, pred_test_labels, true_test_labels = np.zeros((1, tokenizer.vocab_size)), np.zeros((1, tokenizer.vocab_size)), \
        np.zeros(1), np.zeros(1)

    for article_idx, pieces in tqdm(article2piece.items()):
        print(article_idx, len(pieces))
        target_span_list: List[Tuple[int]] = []
        input_y_ids_list = []
        # length of target
        target_len_list = []

        for context, target in pieces:

            if context.strip():
                context: str = "".join(context.split()).replace(
                    "▁", " "
                )  # with whitespace
                target: str = "".join(target.split()).replace(
                    "▁", " "
                )  # with whitespace
                text = context + target
                # has_context_no_space = 0
            else:
                context = "<|endoftext|>"                
                if tokenizer.vocab_size == 16000:
                    context = " "
                target: str = "".join(target.split()).replace(
                    "▁", " "
                )  # no whitespace (first token in sent.)
                text = context + target

            encoded_context: Dict = tokenizer(context, return_tensors="pt")
            encoded_target: Dict = tokenizer(target, return_tensors="pt")
            encoded_text: Dict = tokenizer(text, return_tensors="pt")
            # the length of encoded_context
            context_len: int = len(encoded_context["input_ids"][0])
            target_len: int = len(encoded_target["input_ids"][0])

            target_span: Tuple[int] = (context_len, context_len + target_len - 1)

            target_len_list.append(target_len)
            target_span_list.append(target_span)
            target_span_text = tokenizer.decode(
                encoded_text["input_ids"][0][target_span[0]: target_span[1] + 1]
            )

            assert (target_span_text == target or " " + target_span_text == target)
            input_ids = encoded_text["input_ids"][0]
            input_y_ids_list.append(input_ids)

        batch_size = 4

        num_pieces_added = sum(target_len_list)

        (surprisals_list_with_order_per_article, surprisals_list_per_article), \
            (_pred_test_logits, _pred_test_probs, _pred_test_labels, _true_test_labels), \
            (logits_array_with_order_per_article, probs_array_with_order_per_article,
             true_labels_list_with_order_per_article), \
            _memory_sample_list \
            = \
            get_surprisal_per_article(
                model, input_y_ids_list, target_span_list, target_len_list, loss_fct, device, batch_size,
                num_pieces_added, tokenizer
            )
        assert len(surprisals_list_with_order_per_article) == len(pieces)

        pred_test_logits = np.append(pred_test_logits, _pred_test_logits, axis=0)
        pred_test_probs = np.append(pred_test_probs, _pred_test_probs, axis=0)
        pred_test_labels = np.append(pred_test_labels, _pred_test_labels, axis=0)
        true_test_labels = np.append(true_test_labels, _true_test_labels, axis=0)

        assert len(_memory_sample_list) == num_pieces_added
        assert len(_memory_sample_list) > len(surprisals_list_with_order_per_article)

        logits_dic_all_article[article_idx] = logits_array_with_order_per_article
        probs_dic_all_article[article_idx] = probs_array_with_order_per_article
        true_labels_dic_all_article[article_idx] = true_labels_list_with_order_per_article
        memory_sample_idx_all_article[article_idx] = _memory_sample_list

        surprisal_dic_all_article[article_idx] = surprisals_list_with_order_per_article
        surprisals_list_all_article.extend(surprisals_list_per_article)

    # save
    true_test_labels0 = np.delete(true_test_labels, 0, axis=0)
    pred_test_logits0 = np.delete(pred_test_logits, 0, axis=0)
    pred_test_probs0 = np.delete(pred_test_probs, 0, axis=0)
    pred_test_labels0 = np.delete(pred_test_labels, 0, axis=0)
    true_test_labels0 = [int(i) for i in true_test_labels0]

    # save the results
    if model_name == "gpt2":
        model_name = "gpt2_small"
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
        print(model_name)
    out_dir = dir0 + "surprisals/" + "{}/".format(ngram) + model_name

    os.makedirs(out_dir, exist_ok=True)

    json.dump(surprisal_dic_all_article, open(out_dir + "/surprisals.json", "w"))
    json.dump(
        {"PPL": np.exp(mean(surprisals_list_all_article))},
        open(out_dir + "/PPL_all_articles.txt", "w"),
    )
    json.dump(memory_sample_idx_all_article, open(out_dir + "/memory_sample_idx_all_article.json", "w"))

    os.makedirs(dir0 + 'save_logits/{}'.format(ngram), exist_ok=True)

    np.save(dir0 + 'save_logits/{}/{}_pred_test_logits.npy'.format(ngram, model_name), pred_test_logits0)
    np.save(dir0 + 'save_logits/{}/{}_pred_test_labels.npy'.format(ngram, model_name), pred_test_labels0)
    np.save(dir0 + 'save_logits/{}/{}_pred_test_probs.npy'.format(ngram, model_name), pred_test_probs0)
    np.save(dir0 + 'save_logits/{}/{}_true_test_labels.npy'.format(ngram, model_name), true_test_labels0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("-model_name", type=str, default="gpt2")
    parser.add_argument("-cuda_num", type=int, default="7")
    parser.add_argument("-data_name", default="Brown", type=valid_data_name)

    args = parser.parse_args()

    main(args)
