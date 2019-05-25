import xml.etree.ElementTree as ET
import os
import json
import re
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import config


cwd = config.cwd
root_dir = config.root_dir
senseXml = config.senseXml
json_dict = config.json_dict
txt_dir = config.txt_dir
json_txt = config.json_txt

def parse_dict(file):
    id = 0
    dict_path = open(os.path.join(root_dir, json_dict), 'w')
    dict = {}
    print("parsing Xml file...")
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
    
    if txt_json_exists:
        fileinfo = os.stat(json_txt)
        if fileinfo.st_size > 3000000:
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
    
    exists = os.path.isfile(json_dict)
    if exists:
        fileinfo = os.stat(json_dict)
        if fileinfo.st_size > 95000000:
            print("Annotations dict already exists")
    else:
        parse_dict(senseXml)
    
    dictionary = jsonToDict(dictPath)
    dicttxt = load_txt_to_dict(txt_dir)

lists_list = []
with tqdm(desc="lemma_synset_lists", total=len(dictionary)) as pbar:
    for index in range(len(dictionary)):
        pbar.update(1)
        for key in dictionary[index]:
            if key == "text":
                txt = (str(dictionary[index][key])).lower()
                txt = re.sub(r"[,@\'?\.$%\d:_]", " ", txt, flags=re.I)
                txt = txt.split()
                if key == "annotations":
                    for x in dictionary[index][key]:
                        for y, w in dictionary[index][key][x].items():
                            if y == "anchor":
                                anchor = str(w)
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
                                            txt = [w.replace(anchor, record) for w in txt]
                                        else:
                                            print("not in wordnet")
                                            continue
                                except:
                                    continue
                            else:
                                continue
                else:
                    continue
            lists_list.append(txt)
    # print(lists_list[:100])
return lists_list
