from preprocessing import create_dataset
import multiprocessing
from gensim.models import Word2Vec
from time import time
import os
import config
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("resources_dir", nargs='?', default=config.resources_dir, help="Resources directory path from the root dir")
    parser.add_argument("annotation_dict", nargs='?', default=config.bn2wn_mapping_dict, help="Name of the json dictionary inside the resources folder")
    parser.add_argument("nb_xml", nargs='?', default=2, help="Number of XML files to be used (high-precision and high-coverage), it can be either 1 or 2")
    parser.add_argument("senseXml1", nargs='?', default=config.senseXml1, help="Path to the first xml file")
    parser.add_argument("senseXml2", nargs='?', default=config.senseXml2, help="Path to the second xml file")
    parser.add_argument("bn2wn_mapping_txt", nargs='?', default=config.bn2wn_mapping_txt, help="Name of the txt file that maps wordnet to babelnet senses")

    return parser.parse_args()

class TqdmEpochLogger(CallbackAny2Vec):

     def __init__(self):
         self.tqd = tqdm(desc="epochs", total=config.epochs)

     def on_epoch_begin(self, model):
         self.tqd.update(1)
         self.tqd.refresh()


# Word2Vec Network
def run_training(train_list):

    progress_bar = TqdmEpochLogger()
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=config.min_count,
                         window=config.window,
                         size=config.size,
                         sample=config.sample,
                         alpha=config.alpha,
                         min_alpha=config.min_alpha,
                         negative=config.negative,
                         workers=cores-1,
                         compute_loss=True,
                         callbacks = [progress_bar])
    loss = w2v_model.get_latest_training_loss()

    print("loss ", loss)

    w2v_model.build_vocab(train_list, progress_per=10000)

    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()

    w2v_model.train(train_list, total_examples=w2v_model.corpus_count, epochs=config.epochs, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.wv.save_word2vec_format(os.path.join(config.resources_dir, config.w2v_bin_model_output))

    w2v_model.wv.save_word2vec_format(os.path.join(config.resources_dir, config.w2v_model_output), binary=False)

if __name__ == "__main__":

    args = parse_args()

    exists = os.path.isfile(os.path.join(config.resources_dir, config.final_txt))
    if exists:
        fileinfo = os.stat(os.path.join(config.resources_dir, config.final_txt))
        if fileinfo.st_size > 95000:
            print("final text file already exists loading it from %s..."%os.path.join(config.resources_dir, config.final_txt).split("/")[-1])
            with open(os.path.join(config.resources_dir, config.final_txt),'r') as txtfile:
                training_list = json.load(txtfile)
    else:
        training_list = create_dataset(resources_dir=args.resources_dir, annotation_dict=args.annotation_dict, \
        senseXml1=args.senseXml1, senseXml2=args.senseXml2, nb_xml=args.nb_xml, \
        bn2wn_mapping_txt=args.bn2wn_mapping_txt)

    run_training(training_list)

