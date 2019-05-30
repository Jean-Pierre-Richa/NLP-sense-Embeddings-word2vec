import config
import csv
import json
from gensim.models import KeyedVectors
from tqdm import tqdm
import os
from scipy.stats import spearmanr
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("resources_dir", nargs='?', default=config.resources_dir, help="Resources directory path from the root dir")
    parser.add_argument("gold_file", nargs='?', default=config.tsv_file, help="File containing the gold annotated word similarity")
    parser.add_argument("model_output", nargs='?', default=config.w2v_model_output, help="model output")

    return parser.parse_args()


def load_tsv(gold_file):
    # print(gold_file)
    # print("Loading gold_score file...")
    simlist = []
    with open(gold_file, 'r') as tab_file:
        next(tab_file)
        reader = csv.reader(tab_file, delimiter='\t')
        for word1, word2, word3 in reader:
            w1 = word1.lower()
            w2 = word2.lower()
            gold_score = word3
            simlist.append([w1, w2, gold_score])
    # print(simlist)

    return simlist

def model_output_to_dict(resources_dir, model_output, gold_file):

    senses = KeyedVectors.load_word2vec_format(os.path.join(resources_dir, model_output), binary=False)
    vocab = senses.wv.vocab
    # print(vocab)
    wanted_senses = load_tsv(os.path.join(resources_dir, gold_file))
    # print(wanted_senses)
    tab_dict = {}
    print(len(wanted_senses))
    dict_file = open(os.path.join(resources_dir, config.w2v_output_json), 'w')
    with tqdm(desc="model output to dict", total=len(wanted_senses)) as pbar:
        for senlist in wanted_senses:
            pbar.update(1)
            for sense in senlist:
                # print(sense)
                tab_dict[sense] = sense
                senselist = []
                try:
                    for k in vocab.keys():
                        if (k.split("_")[0] == sense):
                            if k not in tab_dict[sense]:
                                senselist.append(k)
                except:
                    pass
                tab_dict[sense] = senselist
    json.dump(tab_dict, dict_file)
    # print(tab_dict)
    dict_file.close()

    return tab_dict

def word_similarity(resources_dir, gold_file, model_output):

    print(os.path.join(resources_dir, model_output))
    model = KeyedVectors.load_word2vec_format(os.path.join(resources_dir, model_output), binary=False)
    similarity_lists = load_tsv(os.path.join(resources_dir, gold_file))
    exists = os.path.isfile(os.path.join(resources_dir, config.w2v_output_json))
    if exists:
        with open(os.path.join(resources_dir, config.w2v_output_json)) as sensesdict:
            model_senses = json.load(sensesdict)
    else:
        model_senses = model_output_to_dict(resources_dir, model_output, gold_file)

    counter = 0
    simscore = []
    goldscore = []


    for list in similarity_lists:
        similaritylist = [-1]
        word1 = list[0]
        word2 = list[1]
        score = list[2]
        goldscore.append(float(score))

        try:
            sense1 = model_senses[word1]
            # print(sense1)
            sense2 = model_senses[word2]
            # print(sense1)
        except:
            sense1 = []
            sense2 = []

        for v1 in sense1:
            for v2 in sense2:
                similarity = model.wv.similarity(v1, v2)
                similaritylist.append(similarity)
        score = max(similaritylist)
        # print("score ", score)
        simscore.append(score)

    correlation, _ = spearmanr(simscore, goldscore)
    # print(len(simscore), len(goldscore))
    # print(correlation)

    # print("min_count {0}, window {1}, size {2}, sample {3}, alpha {4}, min_alpha {5}, negative {6}, epochs {7}".format(config.min_count, \
    #  config.window, config.size, config.sample, config.alpha, config.min_alpha,\
    #   config.negative, config.epochs))

    return correlation

if __name__ == "__main__":

    args = parse_args()

    _ = word_similarity(resources_dir=args.resources_dir, gold_file=args.gold_file, model_output=args.model_output)
