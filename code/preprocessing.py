import xml.etree.ElementTree as ET
import os
import json
from tqdm import tqdm

cwd = os.getcwd()
root_dir = os.path.join(cwd, '../')
senseXml = os.path.join(root_dir, 'resources/EuroSense-2/eurosense.v1.0.high-precision.xml')
def parse_dict(file):
    id = 0
    dict_path = open(os.path.join(root_dir, 'test.json'), 'w')
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
                annotations["coherenceScore"] = y.attrib["coherenceScore"]
                annotations["babelNet"] = y.text
                dict[id]["annotations"][annotations["lemma"]] = annotations
            y.clear()
    dict_path.close()

if __name__ == "__main__":
    # input list
    parse_dict(senseXml)
