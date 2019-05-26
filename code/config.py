#config
import os

cwd = os.getcwd()

root_dir = os.path.join(cwd, '../')

senseXml = os.path.join(root_dir, 'resources/EuroSense/eurosense.v1.0.high-coverage.xml')

json_dict = os.path.join(root_dir, 'resources/annotations.json')

txt_dir = os.path.join(root_dir, 'resources/bn2wn_mapping.txt')

json_txt = os.path.join(root_dir, 'resources/bn2wn_mapping.json')

tsv_file = os.path.join(root_dir, "resources/wordsim353/combined.tab")

w2v_model_output = os.path.join(root_dir, "resources/modelOutput/model.txt")

w2v_bin_model_output = os.path.join(root_dir, "resources/modelOutput/model.bin")

w2v_output_json = os.path.join(root_dir, "resources/modelOutput/model.json")

# Word2vec

min_count = 5

window    = 5

size      = 100

sample    = 6e-5

alpha     = 0.03

min_alpha = 0.0007

negative  = 10

epochs    = 10
