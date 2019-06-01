# preprocessing
import xml.etree.ElementTree as ET
import os
import json
import re
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import config

resources_dir = config.resources_dir
annotation_dict = config.annotation_dict

'''

    parse the XML files to take the english text and annotations(containing
    the anchor, lemma, and babelNet synset)

'''

def parse_dict(file, aw):
    id = 0
    dict_path = open(os.path.join(resources_dir, annotation_dict), aw)
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
                annotations["babelNet"] = y.text
                dict[id]["annotations"][annotations["lemma"]] = annotations
            y.clear()
    dict_path.close()

'''

    given a json file, load it into a dictionary

'''
def jsonToDict(json_file):

    print("loading json file from %s..."%json_file.split("/")[-1])

    with open(json_file) as fDict:

        final_dict = json.loads("[" +fDict.read().replace("}{", "},{") +"]")

    return final_dict

# load the mapping between babelNet and wordNet text file into a dictionary
def load_txt_to_dict(bn2wn_mapping_txt):

    bn2wn_mapping_dict = os.path.join(resources_dir, config.bn2wn_mapping_dict)
    txt_json_exists = os.path.isfile(bn2wn_mapping_dict)

    if txt_json_exists:
        fileinfo = os.stat(bn2wn_mapping_dict)
        print(fileinfo.st_size)
        if fileinfo.st_size > 3000000:
            print("mapping file already converted into a dict")
    else:
        dict_file = open(bn2wn_mapping_dict, 'w')

        print("loading textfile from %s..."%bn2wn_mapping_txt.split("/")[-1])

        txt_dict = {}
        with open(os.path.join(resources_dir, bn2wn_mapping_txt)) as txtf:
            for line in txtf:
                key, *value = line.split('\t')
                txt_dict[key] = value[0].split('\n')[0]
        json.dump(txt_dict, dict_file)
        dict_file.close()

    print("loading dictionary from %s..."%bn2wn_mapping_dict.split("/")[-1])
    with open(bn2wn_mapping_dict) as txtdict:
        dict_mapping = json.load(txtdict)

    return dict_mapping

'''

     create the dataset (list of lists) from the annotation dictionary (if it exists.
     otherwise parse the xml) containing the text and annotations

'''

def create_dataset(resources_dir, annotation_dict, senseXml1, senseXml2, nb_xml, bn2wn_mapping_txt):

    annotation_dict = os.path.join(resources_dir, config.annotation_dict)
    exists = os.path.isfile(annotation_dict)
    # load the annotation dictionary if it already exists
    if exists:
        fileinfo = os.stat(annotation_dict)
        if fileinfo.st_size > 2100000000:
            print("Annotations dict already exists")
    # else parse the xml file
    else:
        print("parsing xml file 1")
        parse_dict(os.path.join(resources_dir, senseXml1), 'w')
        if nb_xml == 2:
            print("parsing xml file 2")
            parse_dict(os.path.join(resources_dir, senseXml2), 'a')
    # load the annotation json into a dict
    dictionary = jsonToDict(annotation_dict)
    # load the mapping txt into a dict
    dict_mapping = load_txt_to_dict(bn2wn_mapping_txt)

    '''

        for each key text in the dictionary, check the annotations, take the
        anchor search for it in the text, then replace it with the lemma_synset
        tag taking the lemma and synset from the annotations relative to the
        text in the end save the list of lists containing the final sentences
        with the lemma_synset instead of the anchor in a text file for later use

    '''

    lists_list = []
    with tqdm(desc="lemma_synset_lists", total=len(dictionary)) as pbar:
        for index in range(len(dictionary)):
            pbar.update(1)
            for key in dictionary[index]:
                if key == "text":
                    txt = (str(dictionary[index][key])).lower()
                    txt = re.sub("[^a-zA-Z-]+", " ", txt)
                if key == "annotations":
                    for x in dictionary[index][key]:
                        for y, w in dictionary[index][key][x].items():
                            if y == "anchor":
                                if ((' ' + w.lower() + ' ') in txt):
                                    anchor = " " + str(w).lower() + " "
                                else:
                                    continue
                            if y == "lemma":
                                lemma = str(w).lower()
                            if y == "babelNet":
                                try:
                                    if dict_mapping[w]:
                                        bnsynset = str(w)
                                        offset = dict_mapping[w]
                                        synset = wn.synset_from_pos_and_offset(offset[-1], int(offset[:-1]))
                                        if synset:
                                            record = " " + lemma + "_" + bnsynset + " "
                                            txt = txt.replace(anchor, record)
                                        else:
                                            print("not in wordnet")
                                except:
                                    pass
                            else:
                                pass
                else:
                    pass
            txt = txt.split()
            lists_list.append(txt)
    print("Saving the final list of annotated text to %s"%config.final_txt.split("/")[-1])
    with open(os.path.join(resources_dir, config.final_txt), 'w') as finaltxt:
        json.dump(lists_list, finaltxt)

    return lists_list
