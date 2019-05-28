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
# binary model output
w2v_bin_model_output = "modelOutput/model.bin"
# the model output dictionary
w2v_output_json = "modelOutput/model.json"
# the text containing the list of lists of senses
final_txt = "modelOutput/final_text.txt"

# Word2Vec gensim parameters
min_count = 5

window    = 5

size      = 400

sample    = 6e-5

alpha     = 0.03

min_alpha = 0.0007

negative  = 10

epochs    = 5
