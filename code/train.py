# network
from preprocessing import create_dataset
import multiprocessing
from gensim.models import Word2Vec
from time import time
import os
import config
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.model_selection import ParameterGrid
import json
import argparse
from similarity import word_similarity
import gensim

'''

    parse arguments

'''
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--resources_dir",
                        nargs='?',
                        default=config.resources_dir,
                        help="Resources directory path from the root dir")
    parser.add_argument("--annotation_dict",
                        nargs='?',
                        default=config.bn2wn_mapping_dict,
                        help="Name of the json dictionary inside the resources folder")
    parser.add_argument("--senseXml1",
                        nargs='?',
                        default=config.senseXml1,
                        help="Path to the first xml file")
    parser.add_argument("bn2wn_mapping_txt",
                        nargs='?',
                        default=config.bn2wn_mapping_txt,
                        help="Name of the txt file that maps wordnet to babelnet senses")
    parser.add_argument("gold_file",
                        nargs='?',
                        default=config.tsv_file,
                        help="File containing the gold annotated word similarity")
    parser.add_argument("model_output",
                        nargs='?',
                        default=config.w2v_model_output,
                        help="model output")

    return parser.parse_args()

'''

    class to update the epochs progress bar

'''
class TqdmEpoch(CallbackAny2Vec):
    # initialize
     def __init__(self, epoch):
         self.tqd = tqdm(desc="epochs", total=epoch)
    # read from the callback and update the progress bar
     def on_epoch_begin(self, w2v_model):
         self.tqd.update(1)
         self.tqd.refresh()

'''

    class for grid search algorithm

'''
class GridSearch:

    def __init__(self, train_list, grid_params):
        self.params = None
        self.grid_params = grid_params
        self.best_correlation = -2

    def my_rule(word, count, min_count):
        if "_bn:" in word:
            # print(word)
            return gensim.utils.RULE_KEEP
        else:
            return gensim.utils.RULE_DEFAULT

    # Word2Vec Network
    def run_training(self, train_list):

        # looping through the parameters to feed the network using grid search
        for p in ParameterGrid(self.grid_params):

            # self.iter+=1
            progress_bar = TqdmEpoch(p['epochs'])
            cores = multiprocessing.cpu_count()
            w2v_model = Word2Vec(min_count=p['min_count'],
                                 window=p['window'],
                                 size=p['size'],
                                 sample=p['sample'],
                                 alpha=p['alpha'],
                                 min_alpha=p['min_alpha'],
                                 negative=p['negative'],
                                 workers=4,
                                 compute_loss=True,
                                 callbacks = [progress_bar])
            # building the network vocabulary
            w2v_model.build_vocab(train_list, progress_per=10000, trim_rule=GridSearch.my_rule)

            t = time()
            # train the network
            w2v_model.train(train_list, total_examples=w2v_model.corpus_count,
                            epochs=p['epochs'], report_delay=1)
            # save the network output (embeddings) in binary format
            w2v_model.wv.save_word2vec_format(os.path.join(args.resources_dir,
                                              config.w2v_bin_model_output))
            # save the network output in txt format
            w2v_model.wv.save_word2vec_format(os.path.join(args.resources_dir,
                                              args.model_output), binary=False)

            print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

            # call the word similarity function to calculate the scores and
            # correlation of the whole model with respect to the gold dataset
            correlation = word_similarity(resources_dir=args.resources_dir,
                                          gold_file=args.gold_file,
                                          model_output=args.model_output)
            # save the currently calculated correlation if better than the
            # previous best one depending on the parameters used for the training and save the model with a best model tag
            if correlation > self.best_correlation:
                self.best_correlation = correlation
                self.params = p
                w2v_model.wv.save_word2vec_format(os.path.join(config.resources_dir,
                                                  config.w2v_best_model_output), binary=False)

            # keep a log of the performances in a txt file
            performance = "min_count {0}, window {1}, size {2}, sample {3}, alpha {4}, min_alpha {5}, negative {6}, epochs {7}, correlation {8}, xml files {9}".format(
            p['min_count'], p['window'], p['size'], p['sample'], p['alpha'], p['min_alpha'], p['negative'], p['epochs'], \
            correlation)

            print(performance)
            with open(os.path.join(config.resources_dir, config.model_performance), 'a') as mp:
                json.dump(performance, mp)
                mp.write('\n')

if __name__ == "__main__":

    args = parse_args()

    try:
        exists = os.path.isfile(os.path.join(config.resources_dir, config.final_txt))
        # load the final converted text containing the list of lists for the
        # training if it already exists
        if exists:
            fileinfo = os.stat(os.path.join(config.resources_dir, config.final_txt))
            if fileinfo.st_size > 9500000:
                # print("jp2")
                print("final text file already exists loading it from %s..."\
                      %os.path.join(config.resources_dir, config.final_txt).split("/")[-1])
                with open(os.path.join(config.resources_dir, config.final_txt),'r') as txtfile:
                    training_list = json.load(txtfile)

        # else create it
        else:
            training_list = create_dataset(resources_dir=args.resources_dir,
            annotation_dict=args.annotation_dict, senseXml1=args.senseXml1,
            bn2wn_mapping_txt=args.bn2wn_mapping_txt)
    except:
    training_list = create_dataset(resources_dir=args.resources_dir,
                                   annotation_dict=args.annotation_dict,
                                   senseXml1=args.senseXml1,
                                   bn2wn_mapping_txt=args.bn2wn_mapping_txt)

    # create a dict containing the grid search parameters
    grid_params = {'min_count':config.min_count,
                   'window':config.window,
                   'size':config.size,
                   'sample':config.sample,
                   'alpha':config.alpha,
                   'min_alpha':config.min_alpha,
                   'negative':config.negative,
                   'epochs':config.epochs}
    # create an instance of grid search and pass the list of lists and the grid
    # parameters that will be used for the training
    grid = GridSearch(train_list=training_list, grid_params=grid_params)
    # run the training
    grid.run_training(training_list)
