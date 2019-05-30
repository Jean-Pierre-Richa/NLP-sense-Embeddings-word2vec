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

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("resources_dir", nargs='?', default=config.resources_dir, help="Resources directory path from the root dir")
    parser.add_argument("annotation_dict", nargs='?', default=config.bn2wn_mapping_dict, help="Name of the json dictionary inside the resources folder")
    parser.add_argument("nb_xml", nargs='?', default=1, help="Number of XML files to be used (high-precision and high-coverage), it can be either 1 or 2")
    parser.add_argument("senseXml1", nargs='?', default=config.senseXml1, help="Path to the first xml file")
    parser.add_argument("senseXml2", nargs='?', default=config.senseXml2, help="Path to the second xml file")
    parser.add_argument("bn2wn_mapping_txt", nargs='?', default=config.bn2wn_mapping_txt, help="Name of the txt file that maps wordnet to babelnet senses")
    parser.add_argument("gold_file", nargs='?', default=config.tsv_file, help="File containing the gold annotated word similarity")
    parser.add_argument("model_output", nargs='?', default=config.w2v_model_output, help="model output")

    return parser.parse_args()

class TqdmEpoch(CallbackAny2Vec):

     def __init__(self, epoch):
         self.tqd = tqdm(desc="epochs", total=epoch)

     def on_epoch_begin(self, w2v_model):
         self.tqd.update(1)
         self.tqd.refresh()

class gridSearch:

    def __init__(self, train_list, param_grid):
        self.param_grid = param_grid
        self.params = None
        self.best_correlation = -2

    # Word2Vec Network
    def run_training(self, train_list):

        for p in ParameterGrid(self.param_grid):

            # self.iter+=1
            progress_bar = TqdmEpoch(p['epochs'])
            cores = multiprocessing.cpu_count()
            w2v_model = Word2Vec(min_count=p['min_count'],
                                 window=p['window'],
                                 size=p['size'],
                                 sample=config.sample,
                                 alpha=config.alpha,
                                 min_alpha=config.min_alpha,
                                 negative=p['negative'],
                                 workers=cores-1,
                                 compute_loss=True,
                                 callbacks = [progress_bar])

            # loss = w2v_model.get_latest_training_loss()


            w2v_model.build_vocab(train_list, progress_per=10000)

            # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

            t = time()

            w2v_model.train(train_list, total_examples=w2v_model.corpus_count, epochs=p['epochs'], report_delay=1)

            # print("loss ", loss)

            w2v_model.wv.save_word2vec_format(os.path.join(args.resources_dir, config.w2v_bin_model_output))

            w2v_model.wv.save_word2vec_format(os.path.join(args.resources_dir, args.model_output), binary=False)


            print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

            correlation = word_similarity(resources_dir=args.resources_dir, gold_file=args.gold_file, model_output=args.model_output)

            if correlation > self.best_correlation:
                self.best_correlation = correlation
                self.params = p
                w2v_model.wv.save_word2vec_format(os.path.join(config.resources_dir, config.w2v_best_model_output), binary=False)


            performance = "min_count {0}, window {1}, size {2}, sample {3}, alpha {4}, min_alpha {5}, negative {6}, epochs {7}, correlation {8} ".format(p['min_count'], \
            p['window'], p['size'], config.sample, config.alpha, config.min_alpha,\
            p['negative'], p['epochs'], correlation)

            print("starting a new training with: ", performance)
            # performance = "parameters {0}".format(self.param_grid)
            print(performance)
            with open(os.path.join(config.resources_dir, config.model_performance), 'a') as mp:
                json.dump(performance, mp)
                mp.write('\n')

if __name__ == "__main__":

    args = parse_args()

    # exists = os.path.isfile(os.path.join(config.resources_dir, config.final_txt))
    # if exists:
    #     fileinfo = os.stat(os.path.join(config.resources_dir, config.final_txt))
    #     if fileinfo.st_size > 950000:
    #         print("final text file already exists loading it from %s..."%os.path.join(config.resources_dir, config.final_txt).split("/")[-1])
    #         with open(os.path.join(config.resources_dir, config.final_txt),'r') as txtfile:
    #             training_list = json.load(txtfile)
    # else:
    training_list = create_dataset(resources_dir=args.resources_dir, annotation_dict=args.annotation_dict, \
    senseXml1=args.senseXml1, senseXml2=args.senseXml2, nb_xml=args.nb_xml, \
    bn2wn_mapping_txt=args.bn2wn_mapping_txt)

    param_grid = dict(epochs=config.epochs, negative=config.negative, window=config.window, size=config.size, min_count=config.min_count)
    grid = gridSearch(train_list=training_list, param_grid=param_grid)
    grid.run_training(training_list)
