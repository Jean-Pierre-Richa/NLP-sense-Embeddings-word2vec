# similarity
import config
import csv
import json
from gensim.models import KeyedVectors
from tqdm import tqdm
import os
from scipy.stats import spearmanr
import argparse
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot

'''

    parse arguments

'''
# mapping to use booleans in the ArgumentParser
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--resources_dir",
                        nargs='?',
                        default=config.resources_dir,
                        help="Resources directory path from the root dir")
    parser.add_argument("--gold_file",
                        nargs='?',
                        default=config.tsv_file,
                        help="File containing the gold annotated word similarity (directory path from the resources_dir)")
    parser.add_argument("--model_output",
                        nargs='?',
                        default=config.w2v_model_output,
                        help="model output (directory path from the resources_dir)")
    parser.add_argument("--only_senses",
                        type=str2bool,
                        nargs='?',
                        default=True,
                        help="extract from the model output txt only the model senses (can be True or False)")
    parser.add_argument("--draw",
                        type=str2bool,
                        nargs='?',
                        default=True,
                        help="plot vocab (can be True or False)")

    return parser.parse_args()


'''

    function to draw the vocabulary using PCA for dimensionality reduction

'''
def draw_vocab(w2v_model):

    vocab = w2v_model.wv.vocab

    pca_dict = {}

    for sense in list(vocab)[:50]:
        pca_dict[sense] = vocab[sense]
    voc = w2v_model[pca_dict]

    pca = PCA(n_components=2)
    result = pca.fit_transform(voc)

    pyplot.scatter(result[:, 0], result[:, 1])

    words = list(pca_dict)

    for i, word in enumerate(words):
    	pyplot.annotate(word.split('_')[0], xy=(result[i, 0], result[i, 1]))
    pyplot.show()

'''

    extract only the senses of babelNet from the final model output

'''

def keep_model_senses(model_output):

    senses_embed = []
    fv = open(os.path.join(args.resources_dir, config.final_vec), "w")
    with open(model_output, 'r') as mo:
        senses = mo.readlines()
        for line in senses:
            if("_bn:" in line):
                # print(line.split()[0])
                senses_embed.append(line)
            else:
                pass
    first_line = "{} {}\n".format(str(len(senses_embed)), len(senses_embed[0].split())-1)
    fv.write(first_line)
    for sense in senses_embed:
        fv.write(sense)
    fv.close()

'''

    load tab seperated values file containing the gold similarity dataset

'''
def load_tsv(gold_file):
    simlist = []
    with open(gold_file, 'r') as tab_file:
        next(tab_file)
        reader = csv.reader(tab_file, delimiter='\t')
        for word1, word2, word3 in reader:
            w1 = word1.lower()
            w2 = word2.lower()
            gold_score = word3
            simlist.append([w1, w2, gold_score])
    return simlist

'''

    take from the model output file only the vocabulary present in the gold file

'''

def model_output_to_dict(resources_dir, model_output, gold_file):
    # load the senses from the model output
    senses = KeyedVectors.load_word2vec_format(os.path.join(resources_dir,
                                               model_output), binary=False)
    vocab = senses.wv.vocab
    wanted_senses = load_tsv(os.path.join(resources_dir, gold_file))
    tab_dict = {}
    print(len(wanted_senses))
    # create a dictionary containing the model output vocab present in the goldfile
    dict_file = open(os.path.join(resources_dir, config.w2v_output_json), 'w')
    with tqdm(desc="model output to dict", total=len(wanted_senses)) as pbar:
        for senlist in wanted_senses:
            pbar.update(1)
            for sense in senlist:
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

'''

    check word similarities from the model output and the gold dataset and perform the spearman correlation

'''

def word_similarity(resources_dir, gold_file, model_output):
    # load the model output
    model = KeyedVectors.load_word2vec_format(os.path.join(resources_dir,
                                              model_output), binary=False)
    # load the gold dataset
    similarity_lists = load_tsv(os.path.join(resources_dir, gold_file))
    # output dictionary
    exists = os.path.isfile(os.path.join(resources_dir, config.w2v_output_json))
    # load it if it exists
    if exists:
        with open(os.path.join(resources_dir, config.w2v_output_json)) as sensesdict:
            model_senses = json.load(sensesdict)
    else:
        # otherwise, create it
        model_senses = model_output_to_dict(resources_dir, model_output, gold_file)
    # will hold the model output similarity score
    simscore = []
    # will hold the gold dataset score
    goldscore = []
    # take only the words in the gold dataset to search for their respective
    # senses in the model outut and create a list of lists for each one
    for list in similarity_lists:
        similaritylist = [-1]
        word1 = list[0]
        word2 = list[1]
        score = list[2]
        goldscore.append(float(score))

        try:
            sense1 = model_senses[word1]
            sense2 = model_senses[word2]
        except:
            sense1 = []
            sense2 = []

        for v1 in sense1:
            for v2 in sense2:
                similarity = model.wv.similarity(v1, v2)
                similaritylist.append(similarity)
        score = max(similaritylist)
        simscore.append(score)

    correlation, _ = spearmanr(simscore, goldscore)
    print(correlation)
    return correlation

if __name__ == "__main__":

    args = parse_args()
    # read the arguments and call the word_similarity function
    _ = word_similarity(resources_dir=args.resources_dir,
                        gold_file=args.gold_file,
                        model_output=args.model_output)
    # if only the senses are required, then take the model output embeddings and
    # extract only the babelNet senses
    if args.only_senses:
        print("extracting the senses from model output")
        keep_model_senses(os.path.join(args.resources_dir, config.w2v_model_output))
        model_best = KeyedVectors.load_word2vec_format(os.path.join(args.resources_dir, config.final_vec), binary=False)
    else:
        pass
    # test that the saved (senses only) file respects the word2vec format
    model = KeyedVectors.load_word2vec_format(os.path.join(args.resources_dir,
                                                   config.final_vec), binary=False)
    if args.draw:
        draw_vocab(model)
