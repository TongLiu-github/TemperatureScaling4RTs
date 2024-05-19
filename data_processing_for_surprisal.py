import os

import json
import math
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean
import sentencepiece as spm
from collections import defaultdict
from typing import Dict, List, Optional, Tuple






parameter_dic = {
    "Brown":{
        "num_articles": 13,
        "dir": "./PPP_Calculation_Brown/"
    },
    "Natural_Stories":{
    "num_articles": 9,
    "dir": "./PPP_Calculation_Natural_Stories/"
    }
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def valid_data_name(value):
    valid_values = ["Brown", "Natural_Stories", "Dundee"]
    if value not in valid_values:
        raise argparse.ArgumentTypeError(f"Invalid value for -data_name. Allowed values are {', '.join(valid_values)}.")
    return value

def main(args):
    
    num_article = parameter_dic[args.data_name]["num_articles"]
    dir0 = parameter_dic[args.data_name]["dir"]
    
#     !wget https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt
    sp = spm.SentencePieceProcessor()
    sp.Load("./models/spm_en/en_wiki.model")
    
    
    with open(dir0 + 'data/all_stories_dict.json', 'r') as json_file:
        words_dic = json.load(json_file)
    
    # bunsetsu_list

    processed_all_stories_dict = {}
    ngram = args.n

    def extract_subwords(token_list):
        return " ".join(c for c in token_list)

    def attach_context(tokens_list):

        return [
            [
            extract_subwords(tokens_list[max(0, idx - ngram + 1) : idx]),
            extract_subwords([b])
            ]
            for idx, b in enumerate(tokens_list)
        ]

    # we have 13 stories in total.
    for i in range(1, num_article + 1):
#         section_list: List[Tuple[str, str]] = []
        section_list = []
        print(i)
        words_list = words_dic[str(i)]

        out = attach_context(words_list)

        # encode using spm
        for idx, [context, gold] in enumerate(out):
            if context == '':
                _context = [context]
                _gold = sp.EncodeAsPieces(gold)
            else:
                _context = sp.EncodeAsPieces(context)
                _gold = sp.EncodeAsPieces(gold)

            section_list.append([" ".join(_context), " ".join(_gold)])

        processed_all_stories_dict[str(i)] = section_list

    
    # save the data
    filename = "processed_all_stories_dict"
    
    folder_path = dir0 + "data/{}".format(ngram)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    
    json.dump(processed_all_stories_dict, open(dir0 + "data/{}/{}.json".format(ngram, filename), "w"), ensure_ascii=False)
    

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_name", default="Brown", type=valid_data_name)
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()
    
    set_seed(4242424242)
    
    main(args)