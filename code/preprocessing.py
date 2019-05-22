import xml.etree.ElementTree as ET
import os
import json
from tqdm import tqdm
from nltk.corpus import wordnet as wn

cwd = os.getcwd()
root_dir = os.path.join(cwd, '../')
senseXml = os.path.join(root_dir, 'resources/EuroSense-2/eurosense.v1.0.high-precision.xml')
json_dict = os.path.join(root_dir, 'resources/annotations.json')
txt_dir = os.path.join(root_dir, 'resources/bn2wn_mapping.txt')
json_txt = os.path.join(root_dir, 'resources/bn2wn_mapping.json')


def parse_dict(file):
    id = 0
    dict_path = open(os.path.join(root_dir, json_dict), 'w')
    dict = {}

    iteration = ET.iterparse(file)
    x, y = iter(iteration).__next__()
    with tqdm(desc="XML->dict", total=len(file)) as pbar:
        for x, y in iter(iteration):
            if y.tag == 'sentence':
                if ((len(list(dict.items())) > 1)):
                    item = list(dict.items())[-1]
                    json.dump(dict[id], dict_path)
                    dict.clear()
                pbar.update(1)
                id = y.attrib["id"]
                dict[id] = {}
                dict[id]["id"] = id
            elif y.tag == "text" and y.attrib["lang"] == "en" and int(id) > 0:
                dict[id]["text"] = y.text
                dict[id]["annotations"] = {}
            elif y.tag == 'annotation' and y.attrib["lang"] == 'en' and int(id) > 0:
                annotations = {}
                annotations["anchor"] = y.attrib["anchor"]
                annotations["lemma"] = y.attrib["lemma"]
                # annotations["coherenceScore"] = y.attrib["coherenceScore"]
                annotations["babelNet"] = y.text
                dict[id]["annotations"][annotations["lemma"]] = annotations
            y.clear()
    dict_path.close()

def jsonToDict(json_file):

    print("loading json file from %s..."%json_file.split("/")[-1])

    with open(json_file) as fDict:

        final_dict = json.loads("[" +fDict.read().replace("}{", "},{") +"]")

    return final_dict

def load_txt_to_dict(txtPath):

    txt_json_exists = os.path.isfile(json_txt)
    if txt_json_exists and fileinfo.st_size > 30000000:
        print("mapping file already converted into a dict")
    else:
        dict_file = open(os.path.join(root_dir, json_txt), 'w')

        print("loading textfile from %s..."%txtPath.split("/")[-1])

        txt_dict = {}
        with open(txtPath) as txtf:
            for line in txtf:
                key, *value = line.split('\t')
                txt_dict[key] = value

        json.dump(txt_dict, dict_file)
        dict_file.close()

    print("loading dictionary from %s..."%json_txt.split("/")[-1])
    with open(json_txt) as txtdict:
        dicttxt = json.load(txtdict)

    return dicttxt

def create_dataset(dictPath):

    dictionary = jsonToDict(dictPath)
    dicttxt = load_txt_to_dict(txt_dir)

    lists_list = []
    with tqdm(desc="lemma_synset_lists", total=len(dictionary)) as pbar:
        for index in range(len(dictionary)):
            pbar.update(1)
            for key in dictionary[index]:
                if key == "annotations":
                    lemma_syn_list = []
                    for x in dictionary[index][key]:
                        for y, w in dictionary[index][key][x].items():
                            if y == "lemma":
                                lemma = str(w)
                            if y == "babelNet":
                                try:
                                    if dicttxt[w]:
                                        bnsynset = str(w)
                                        offset = dicttxt[w][0].split("\n")[0]
                                        synset = wn.synset_from_pos_and_offset(offset[-1], int(offset[:-1]))
                                        if synset:
                                            record = lemma + "_" + bnsynset
                                            lemma_syn_list.append(record)
                                        else:
                                            print("not in wordnet")
                                            continue
                                except:
                                    continue
                    lists_list.append(lemma_syn_list)
                else:
                    continue
    return lists_list

if __name__ == "__main__":
    exists = os.path.isfile(json_dict)
    if exists:
        fileinfo = os.stat(json_dict)
        if fileinfo.st_size > 950000000:
            print("Annotations dict already exists")
    else:
        parse_dict(senseXml)

    create_dataset(json_dict)
