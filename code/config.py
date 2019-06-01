#config
import os

# Get current working directory
cwd = os.getcwd()
# root folder
root_dir = os.path.join(cwd, '../')
# resources folder
resources_dir = os.path.join(root_dir, "resources/")
# XML files
senseXml1 = 'EuroSense/eurosense.v1.0.high-coverage.xml'
senseXml2 = 'EuroSense-2/eurosense.v1.0.high-precision.xml'
# the dictionary that will hold the data after preprocessing the XML files
annotation_dict = 'annotations.json'
# the file that contains the mapping between wordnet and babelNet
bn2wn_mapping_txt = 'bn2wn_mapping.txt'
# the dictionary that is created to contain the mapping between wordnet and babelNet
bn2wn_mapping_dict = 'bn2wn_mapping.json'
# the tab seperated value gold dataset
tsv_file = "wordsim353/combined.tab"
# text model output in Wor2Vec format
w2v_model_output = "modelOutput/model.txt"
# text best model output in Word2Vec format
w2v_best_model_output = "modelOutput/best_model.txt"
# binary model output
w2v_bin_model_output = "modelOutput/model.bin"
# the model output dictionary
w2v_output_json = "modelOutput/model.json"
# the text containing the list of lists of senses
final_txt = "modelOutput/final_text.txt"
# text file saving all the training parameters and correlation
model_performance = "modelOutput/models_performance.txt"
# final best embeddings file
final_vec = "modelOutput/embeddings.txt"
# Word2Vec gensim parameters
# words that will be considered following a minimum count in the corpus
min_count = [5, 10]
# window of words before and after the predicted word
window    = [10, 5]
# embeddings size
size      = [400, 200]

sample    = [0.00006, 0.00001]
# starting learning rate
alpha     = [0.03, 0.01]
# ending learning rate
min_alpha = [0.0007, 0.0002]
# negative sampling
negative = [10, 10]
# number of epochs
epochs = [30, 20]
