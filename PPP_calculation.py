import os
import sys
import json
import glob
import torch
import random
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from statistics import mean
from scipy.stats import entropy
from scipy.special import softmax

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


def surprisal_calculate(probs, true_class):
    return -1 * torch.log2(torch.tensor(probs[true_class]))




def generate_surprisals(memory_sample_idx_all_article, cal_test_probs0, true_test_labels0, num_articles):
    true_test_labels0 = [int(i) for i in true_test_labels0]

    surprisal_rst_all_article = {}

    # get the article length (including punctuation) list
    i_article_len_list = []
    for i in range(1, num_articles + 1):
        i_article_len = len(memory_sample_idx_all_article[str(i)])
        i_article_len_list.append(i_article_len)

    for article_idx in tqdm(range(1, num_articles + 1)):
        # start from the first article

        if article_idx == 1:
            start, end = 0, i_article_len_list[article_idx - 1]
        else:
            start, end = end, end + i_article_len_list[article_idx - 1]

        # (including punctuation)
        i_article_len = i_article_len_list[article_idx - 1]

        # get the first article's logits
        #         pred_test_logitsi = pred_test_logits0[start : end, :]
        #         pred_test_probsi = softmax(pred_test_logitsi, -1)
        cal_test_probsi = cal_test_probs0[start: end, :]
        true_test_labelsi = true_test_labels0[start: end]

        # calculate surprisal
        surprisal_listi = []
        for i in range(i_article_len):
            _surprisal = surprisal_calculate(cal_test_probsi[i], true_test_labelsi[i])
            #     print(_surprisal)
            _surprisal = _surprisal.item()
            surprisal_listi.append(_surprisal)

        assert len(surprisal_listi) == i_article_len

        # get surprisals with order and without summing punctation
        memory_sample_idxi = memory_sample_idx_all_article[str(article_idx)]
        memory_sample_idx_arrayi = np.asarray(memory_sample_idxi)

        # sort
        sort_idx = np.argsort(memory_sample_idx_arrayi[:, 0])
        sorted_memory_sample_idx_arrayi = memory_sample_idx_arrayi[sort_idx]

        sorted_surprisal_arrayi = np.asarray(surprisal_listi)[sort_idx]

        # merge the punctuation
        sorted_surprisal_listi = []
        skip_loop_idx = 0
        for idx, (i, j) in enumerate(zip(sorted_surprisal_arrayi, sorted_memory_sample_idx_arrayi)):
            if j[1] == 1:
                sorted_surprisal_listi.append(i)
            elif j[1] > 1 and skip_loop_idx == 0:
                # skip two times if j[1] == 3
                skip_loop_idx = j[1] - 1
                
                # Here I average the surprisals after scaling for subwords/puctuations
                _s = sum(sorted_surprisal_arrayi[idx: idx + j[1]])
                #                 _s = sum(sorted_surprisal_arrayi[idx: idx + j[1]]) / j[1]
                sorted_surprisal_listi.append(_s)
                # skip j[1] times loop
            elif j[1] > 1 and skip_loop_idx > 0:
                skip_loop_idx -= 1
                continue

        # restore
        surprisal_rst_all_article[str(article_idx)] = sorted_surprisal_listi

    return surprisal_rst_all_article


def concat_results(surprisals, article_list, prev: int = -1):
    if prev > 0:
        results = []
        for idx in article_list:
            avg_surprisal_per_article = mean(surprisals[idx])
            results.extend([avg_surprisal_per_article] * prev + surprisals[idx][: -1 * prev])
        return results
    else:
        # return each article's ordered surprisals in one loop, then for 10 loops (each surprisal return 10 times)
        return [s for a in article_list for s in surprisals[a]]


def get_PPP(surprisals, model_name, K, args, input_dir, dir0):
    with open(dir0 + "data/article_order.txt") as f:
        article_list = [a.strip() for a in f]
    f.close()

    _dir = input_dir + model_name + "_" + K + "_" + "scores.csv"
    print(_dir)
    with open(_dir, "w") as f:
        # header "surprisals_sum(_prev_1/2/3)"
        header = "\t".join(
            [
                "surprisals_sum" if prev == 0 else f"surprisals_sum_prev_{prev}"
                for prev in range(0, 4)
            ]
        )
        f.write(header + "\n")
        scores = {}
        scores["surprisals_sum"] = concat_results(surprisals, article_list)
        for i in range(1, 4):
            scores["surprisals_sum_prev_" + str(i)] = concat_results(
                surprisals, article_list, prev=i
            )

        # so beautiful !
        f.write(
            "\n".join(
                [
                    "\t".join([str(scores[h][i]) for h in header.split("\t")])
                    for i in range(len(scores["surprisals_sum"]))
                ]
            )
            + "\n"
        )


def get_avg(surprisal_list, num_articles):
    total = 0
    avg = []
    for idx in range(1, num_articles + 1):

        total += len(surprisal_list[str(idx)])

        for xl2 in surprisal_list[str(idx)]:
            avg.append(xl2)
    avg = sum(avg) / total
    return avg


def get_entropy(surprisal_list):
    _surprisal_list = []
    for idx in range(1, 21):
        _surprisal_list.extend(surprisal_list[str(idx)])
    return entropy(_surprisal_list, base=2)


def main(args):
    model_name = args.model_name
    if model_name == "gpt2":
        model_name = "gpt2_small"

    if "/" in model_name:
        model_name = model_name.split("/")[-1]
        print(model_name)

    num_articles = parameter_dic[args.data_name]["num_articles"]
    dir0 = parameter_dic[args.data_name]["dir"]
    ngram = args.n

    pred_test_logits = np.load(dir0 + 'save_logits/{}/{}_pred_test_logits.npy'.format(args.n, model_name))
    pred_test_labels = np.load(dir0 + 'save_logits/{}/{}_pred_test_labels.npy'.format(args.n, model_name))
    pred_test_probs = np.load(dir0 + 'save_logits/{}/{}_pred_test_probs.npy'.format(args.n, model_name))
    true_test_labels = np.load(dir0 + 'save_logits/{}/{}_true_test_labels.npy'.format(args.n, model_name))
    print(pred_test_logits.shape, )
    true_test_labels = [int(i) for i in true_test_labels]
    pred_test_labels = [int(i) for i in pred_test_labels]
    pred_test_logits0 = torch.from_numpy(pred_test_logits)
    pred_test_probs0 = torch.from_numpy(pred_test_probs)

    # Open the JSON file
    input_dir = dir0 + "surprisals/" + "{}/".format(ngram) + model_name + "/"
    with open(input_dir + 'memory_sample_idx_all_article.json', 'r') as f:
        # Load the JSON data
        memory_sample_idx_all_article = json.load(f)
    f.close()

    if args.K == 10:
        T_list = [args.T_optimal]

    elif args.K == 0:
        T_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.25, 2.5, 2.75,
                  3., 3.25, 3.5, 4., 4.5, 5.,
                  5.5, 6., 6.5, 7., 8., 9., 10.]

    ece_list, PPP_list = [], []

    for T in tqdm(T_list):

        pred_test_logits1 = pred_test_logits0 / T
        cal_test_probs1 = pred_test_logits1.softmax(-1)
        cal_test_probs = cal_test_probs1.numpy()
        surprisal_rst_all_article = generate_surprisals(memory_sample_idx_all_article, cal_test_probs, true_test_labels, \
                                                        num_articles)
        K = args.K
        get_PPP(surprisal_rst_all_article, model_name, str(K), args, input_dir, dir0)
        if args.data_name == "Brown":
            R_name = "brown"
        elif args.data_name == "Natural_Stories":
            R_name = "natural"
        subprocess.run(
            ['Rscript', '{}{}.r'.format(dir0, R_name), dir0 + 'surprisals/{}'.format(args.n), model_name, str(K), dir0])

        # read the PPP
        text_dir = input_dir + model_name + "_" + str(K) + "_" + " PPP.txt"
        with open(text_dir) as f:
            lines = f.readlines()
        PPP = float(lines[1].split(' ')[-1][:-2])

        PPP_list.append(PPP)

    PPP_dir = input_dir + model_name + "_" + "_" + " PPP_result" + str(K) + ".txt"
    with open(PPP_dir, "w") as f:
        f.write(str(PPP_list))
        f.write(str(T_list))
        f.write(str(ece_list))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("-model_name", type=str, default="gpt2")
    parser.add_argument("-cuda_num", type=int, default="0")
    parser.add_argument("-data_name", type=str, default='Brown')
    parser.add_argument("-K", type=int, default=10)
    parser.add_argument("-T_optimal", type=float, default=1.)

    args = parser.parse_args()

    set_seed(4242424242)

    main(args)
