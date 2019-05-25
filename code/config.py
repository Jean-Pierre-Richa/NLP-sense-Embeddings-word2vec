#config
import os

cwd = os.getcwd()

root_dir = os.path.join(cwd, '../')

senseXml = os.path.join(root_dir, 'resources/EuroSense-2/eurosense.v1.0.high-precision.xml')

json_dict = os.path.join(root_dir, 'resources/annotations.json')

txt_dir = os.path.join(root_dir, 'resources/bn2wn_mapping.txt')

json_txt = os.path.join(root_dir, 'resources/bn2wn_mapping.json')

# Word2vec

min_count = 20

window    = 2

size      = 100

sample    = 6e-5

alpha     = 0.03

min_alpha = 0.0007

negative  = 20

epochs    = 20
