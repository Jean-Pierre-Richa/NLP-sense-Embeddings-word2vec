from preprocessing import create_dataset
import multiprocessing
from gensim.models import Word2Vec
from time import time
import os
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot
import config
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec


json_dict = config.json_dict

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
                         callbacks = [progress_bar])
    t = time()

    w2v_model.build_vocab(train_list, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()

    w2v_model.train(train_list, total_examples=w2v_model.corpus_count, epochs=config.epochs, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.wv.save_word2vec_format(config.w2v_bin_model_output)

    w2v_model.wv.save_word2vec_format(config.w2v_model_output, binary=False)

    # X = w2v_model[w2v_model.wv.vocab]

    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    #
    # pyplot.scatter(result[:, 0], result[:, 1])
    #
    # words = list(w2v_model.wv.vocab)
    #
    # for i, word in enumerate(words):
    # 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()

if __name__ == "__main__":

    training_list = create_dataset(json_dict)

    run_training(training_list)
